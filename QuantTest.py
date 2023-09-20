import sys
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data
import torch.onnx

from torch import nn
import numpy as np

import torch.utils.benchmark as benchmark

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models
from torchsummary import summary

import gc

sys.path.append("./references/classification/")

# add quantizer model layers, these layers will be subsituted with quantized version
# using initialize before any models are loaded will cause it to automatically happen.
from pytorch_quantization import quant_modules
quant_modules.initialize()

from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.densenet import DenseNet161_Weights

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
CPU_MAX_BATCH_SIZE = 32
ACCURACY_COLUMN_NAME = "Accuracy"
DATA_DIR = '/mnt/d/DataSets/imagenet/'
OUT_DIR = './output/'
N_CHANNELS = 3
IMG_SIZE=224
MAX_BATCH_SIZE=BATCH_SIZES[-1]

model_name = 'ResNet-50'
model = models.resnet50(pretrained=True)
preprocess = ResNet50_Weights.DEFAULT.transforms()

# model_name = 'DenseNet-161'
# model = models.densenet161(pretrained=True)
# preprocess = DenseNet161_Weights.DEFAULT.transforms()

def get_dataloader_subset(dataset: datasets.ImageNet, batch_size: int, seed: int = 42):
    global preprocess


    # Select 20% of the dataset
    subset_fraction = 0.2  
    subset_size = int(len(dataset) * subset_fraction)

    # Create a subset of the larger dataset.
    torch.manual_seed(seed) 
    indices_shuffled = torch.randperm(len(dataset))
    subset_indices = indices_shuffled[:subset_size]
    subset_dataset = Subset(dataset, subset_indices)

    # weights = ResNet50_Weights.DEFAULT
    # preprocess = weights.transforms()

    dataloader = DataLoader(
        subset_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: (torch.stack([preprocess(item[0]) for item in batch]), [item[1] for item in batch]))
    
    return dataloader

def collect_stats(model, data_loader, num_batches=1):
   
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
        
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break
    
    # disable calibrators
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

def evaluate(model, data_loader, device) -> None:
    
    model.eval()

    # Lists to store true and predicted labels
    true_preds = []
    all_preds = []
       
    for data, labels in data_loader:

        data = data.to(device)

        # Forward pass through the model
        outputs = model(data)
        
        # Compute predicted labels (e.g., argmax for classification)
        _, predicted = torch.max(outputs, 1)
        
        # Collect true and predicted labels
        true_preds.extend(labels)
        all_preds.extend(predicted.to('cpu').tolist())

    # Compute accuracy by comparing true and predicted labels
    accuracy = 100 * (np.array(true_preds) == np.array(all_preds)).mean()
    print(f"Accuracy: {accuracy}%")
    print("")

def QuantizationMain():
    global model

    # Post Quantization Training

    # Calibration - collect statistics (fixed range) for quantization based on model.
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    # model = models.resnet50(pretrained=True)
    model.cuda()

    # Use a different seed than what was used during 8 bit quantization.
    dataset = datasets.ImageNet(DATA_DIR, split='val')
    data_loader = get_dataloader_subset(dataset, CPU_MAX_BATCH_SIZE, seed=99) 

    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, data_loader, num_batches=2)
        compute_amax(model, method="percentile", percentile=99.99)

    # Evaluate the calibrated model
    with torch.no_grad():
        evaluate(model, data_loader, device="cuda")

    torch.save(model.state_dict(), OUT_DIR + f'quant-{model_name}-calibrated.pth')

    # data_loader = get_dataloader_subset(dataset, CPU_MAX_BATCH_SIZE, seed=100) 
    # with torch.no_grad():
    #     compute_amax(model, method="percentile", percentile=99.9)
    #     evaluate(model, data_loader, device="cuda")

    # data_loader = get_dataloader_subset(dataset, CPU_MAX_BATCH_SIZE, seed=101) 
    # with torch.no_grad():
    #     for method in ["mse", "entropy"]:
    #         print(F"{method} calibration")
    #         compute_amax(model, method=method)
    #         evaluate(model, data_loader, device="cuda")



def inference_test(model: nn.Module, input_data) -> None:
    with torch.no_grad():
        _ = model(input_data)

def run_speed_benchmark_tests(
        model: nn.Module, 
        df: pd.DataFrame) -> None:
        
    batch_size_subset = BATCH_SIZES
   
    model.eval()
    model_index = len(df) # Store the index for updating while datasets run. 
    df.loc[model_index] = [model_name] + ['NVIDIA 8 Bit Quantization'] + [0.0] * len(batch_size_subset)

    for batch_size in batch_size_subset:

        # Generate fuzzed data for the inputs
        input_shape = (batch_size, N_CHANNELS, IMG_SIZE, IMG_SIZE)
        
        fuzzer = benchmark.Fuzzer(
                parameters = [],
                tensors = [
                    benchmark.FuzzedTensor(
                        'x', 
                        size=input_shape, 
                        dtype=torch.float32, 
                        cuda=True,
                        probability_contiguous=0.6)            
                    
                ],
                seed=0,
            )
        
        for tensors, _, _ in fuzzer.take(1):
            input_data = tensors["x"]
            
        # Environment Setup
        vars = {
            'inference_test': inference_test, 
            'model_item': model, 
            'input_data': input_data
        }

        # Benchmark returns the time per run and handles:
        #   Cuda Synchronization
        #   warm up runs
        #   multiple iterations
        benchmark_timer = benchmark.Timer(
            stmt='inference_test(model_item, input_data)',
            globals=vars,
            num_threads=1)

        # Get the measurement results from the benchmark
        benchmark_result = benchmark_timer.blocked_autorange()

        df.at[model_index, f"Batch Size {batch_size} (ms)"] = benchmark_result.median * 1000   # Convert to milliseconds

        print(f"{model_name}-{'NVIDIA 8 Bit Quantization'} ({batch_size} Batch) Time: {benchmark_result.median * 1000} ms")
        print("")

        csv_path = f'{OUT_DIR}temp_speed_results.csv'
        df.to_csv(csv_path, index=False)

        gc.collect() # This may not be necessary, but due to some segmentation faults, this is here just in case for now. 
    
    print(f"Ending Speed Test for: {model_name}-{'NVIDIA 8 Bit Quantization'}")

def SpeedRunMain():
    global model

    batch_size_subset = BATCH_SIZES
    columns = ["Model", "Options"] + [f"Batch Size {batch_size} (ms)" for batch_size in batch_size_subset]
    df = pd.DataFrame(columns=columns)

    # model = models.resnet50(pretrained=True)
    model.cuda()
    model.load_state_dict(torch.load(OUT_DIR + f'quant-{model_name}-calibrated.pth'))
   
    dummy_input = torch.randn(1, N_CHANNELS, IMG_SIZE, IMG_SIZE).to(device='cuda')  # Batch size 1, 3 channels, 224x224 image size
    torch.onnx.export(model, dummy_input, OUT_DIR + f'quant-{model_name}-calibrated.onnx', verbose=True, input_names=['input'], output_names=['output'])
   
   
   #  summary(model, (N_CHANNELS, IMG_SIZE, IMG_SIZE))

    run_speed_benchmark_tests(model, df)

    
    df.to_csv(f'{OUT_DIR}{model_name}_NVIDIA_8_Bit_Quantization_speed_results.csv', index=False)

if __name__ == "__main__":
   
    #model.cuda()
    # summary(model, (N_CHANNELS, IMG_SIZE, IMG_SIZE))
    #torch.save(model.state_dict(), OUT_DIR + f'{model_name}-raw.pth')

    #model.cuda()
    #dummy_input = torch.randn(1, N_CHANNELS, IMG_SIZE, IMG_SIZE).to(device='cuda')  # Batch size 1, 3 channels, 224x224 image size
    #torch.onnx.export(model, dummy_input, OUT_DIR + f'{model_name}-raw.onnx', verbose=True, input_names=['input'], output_names=['output'])
    
    # QuantizationMain()
    SpeedRunMain()
