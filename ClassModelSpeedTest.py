import copy
from typing import List
from collections import namedtuple
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch.utils.data import DataLoader, Subset

import torch_tensorrt
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.densenet import DenseNet161_Weights
from torchvision.models.squeezenet import SqueezeNet1_1_Weights
from torchvision.models.convnext import ConvNeXt_Base_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights

from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

import gc

ModelInfo = namedtuple("ModelInfo", "name options model FP32_precision_level FP16_precision INT8_precision")

# Define different batch sizes to test
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
CPU_MAX_BATCH_SIZE = 32
ACCURACY_COLUMN_NAME = "Accuracy"
DATA_DIR = '/mnt/d/DataSets/imagenet/'
OUT_DIR = './output/'
N_CHANNELS = 3
IMG_SIZE=224
MAX_BATCH_SIZE=BATCH_SIZES[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(OUT_DIR)
    if not isExist:
        os.makedirs(OUT_DIR)
    
    # Define the models and their corresponding weights
    model_configs = [
        (models.convnext_base, ConvNeXt_Base_Weights.DEFAULT),
        (models.densenet161, DenseNet161_Weights.DEFAULT),
        (models.resnet50, ResNet50_Weights.IMAGENET1K_V2),
        (models.squeezenet1_1, SqueezeNet1_1_Weights.DEFAULT),
        (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT)
    ]

    # Perform Speed Test on GPU for FP32, FP16
    gpu_models = [model(weights=weight) for model, weight in model_configs]
    run_speed_benchmark_tests_by_type(gpu_models, 'gpu')
    # run_accuracy_benchmark_tests_by_type(gpu_models, 'gpu')

    # Perform Speed Test on CPU for FP32, FP16
    global device
    device = torch.device('cpu')
    cpu_models = [model(weights=weight) for model, weight in model_configs if model not in [models.resnet50, models.efficientnet_b0]]
    run_speed_benchmark_tests_by_type(cpu_models, 'cpu')
    # run_accuracy_benchmark_tests_for_cpu(cpu_models)

def run_speed_benchmark_tests_by_type(models: List[nn.Module], testing_mode: str) -> None:
    assert ((testing_mode == 'gpu' and device.type == 'cuda') or \
            (testing_mode == 'cpu' and device.type == 'cpu'))

    batch_size_subset = BATCH_SIZES
    if testing_mode == 'cpu':
        batch_size_subset = BATCH_SIZES[:BATCH_SIZES.index(CPU_MAX_BATCH_SIZE) + 1]

    for model in models:
        model_name = type(model).__name__
        
        model_list = create_model_optimization_records_gpu(model) if testing_mode == 'gpu' else create_model_optimization_records_cpu(model)
        
        columns = ["Model", "Options"] + [f"Batch Size {batch_size} (ms)" for batch_size in batch_size_subset]
        df = pd.DataFrame(columns=columns)

        run_speed_benchmark_tests(model_list, df)

        csv_path = f'{OUT_DIR}{model_name}_{testing_mode}_speed_results.csv'
        df.to_csv(csv_path, index=False)

        graph_path = f'{OUT_DIR}{model_name}_{testing_mode}_speed_results.png'
        CreateBarGraph(model_name, csv_path, graph_path)


def run_accuracy_benchmark_tests_by_type(models: List[nn.Module], testing_mode: str) -> None:
    assert ((testing_mode == 'gpu' and device.type == 'cuda') or \
            (testing_mode == 'cpu' and device.type == 'cpu'))

    for model in models:
        model_name = type(model).__name__
        model_list = create_model_optimization_records_gpu(model) if testing_mode == 'gpu' else create_model_optimization_records_cpu(model)

        columns = ["Model", "Options", ACCURACY_COLUMN_NAME]

        df = pd.DataFrame(columns=columns)

        run_accuracy_benchmark_tests(model_list, df)

        csv_path = f'{OUT_DIR}{model_name}_{testing_mode}_accuracy_results.csv'
        df.to_csv(csv_path, index=False)

        graph_path = f'{OUT_DIR}{model_name}_{testing_mode}_accuracy_results.png'
        CreateBarGraph(model_name, csv_path, graph_path)
    
''' Create the settings to drive the speed tests for GPU related inference. 

    This was originally created in a loop, however as different settings were 
    being changed the loop became complex and listing each configuration made more sense.'''
def create_model_optimization_records_gpu(standard_model: nn.Module) -> List[ModelInfo]:
    # This function is only for GPU testing
    assert(device.type == 'cuda')

    result: List[ModelInfo] = []
    model_name: str = type(standard_model).__name__
    
    standard_model.to(device)
    standard_model.eval()
    
    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "Uncompiled (highest precision)",                   # options
        copy.deepcopy(standard_model),                      # model
        "highest",                                          # FP32_precision
        False,                                              # FP16_precision
        False))                                             # INT8_precision

    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "Compiled (highest precision)",                     # options
        torch.compile(copy.deepcopy(standard_model)),       # model
        "highest",                                          # FP32_precision
        False,                                              # FP16_precision
        False))                                             # INT8_precision

    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "Compiled (high precision)",                        # options
        torch.compile(copy.deepcopy(standard_model)),       # model
        "high",                                             # FP32_precision
        False,                                              # FP16_precision
        False))                                             # INT8_precision

    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "Compiled (medium precision)",                      # options
        torch.compile(copy.deepcopy(standard_model)),       # model
        "medium",                                           # FP32_precision
        False,                                              # FP16_precision
        False))                                             # INT8_precision

    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "Compiled (16 bit precision)",                      # options
        torch.compile(copy.deepcopy(standard_model).half()),# model
        None,                                               # FP32_precision
        True,                                               # FP16_precision
        False))                                             # INT8_precision
   
    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "NVIDIA TensorRT (32 bit)",                         # options
        trt_compile(copy.deepcopy(standard_model)),         # model
        None,                                               # FP32_precision
        False,                                              # FP16_precision
        False))                                             # INT8_precision
    
    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "NVIDIA TensorRT (16 bit)",                         # options
        trt_compile(copy.deepcopy(standard_model).half(), True), # model
        None,                                               # FP32_precision
        True,                                               # FP16_precision
        False))                                             # INT8_precision
   
    return result

''' Create the settings to drive the speed tests for CPU related inference. 

    This was originally created in a loop, however as different settings were 
    being changed the loop became complex and listing each configuration made more sense.'''
def create_model_optimization_records_cpu(standard_model: nn.Module) -> List[ModelInfo]:
    # This function is only for CPU testing
    assert(device.type == 'cpu')
    
    result: List[ModelInfo] = []
    model_name: str = type(standard_model).__name__
    
    standard_model.to(device)
    standard_model.eval()
    
    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "Uncompiled (highest precision)",                   # options
        copy.deepcopy(standard_model),                      # model
        "highest",                                          # FP32_precision
        False,                                              # FP16_precision
        False))                                             # INT8_precision

    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "Compiled (highest precision)",                     # options
        torch.compile(copy.deepcopy(standard_model)),       # model
        "highest",                                          # FP32_precision
        False,                                              # FP16_precision
        False))                                             # INT8_precision

    result.append(ModelInfo(
        f"{model_name}",                                    # name
        "Pytorch Quantized (8 Bit)",                        #options
        eightbit_compile(copy.deepcopy(standard_model)),  # model
        None,                                               # FP32_precision
        False,                                              # FP16_precision
        True))                                              # INT8_precision

    return result

def run_speed_benchmark_tests(
        model_list: List[ModelInfo], 
        df: pd.DataFrame) -> None:
        
    batch_size_subset = BATCH_SIZES
    if device.type == 'cpu':
        batch_size_subset = BATCH_SIZES[:BATCH_SIZES.index(CPU_MAX_BATCH_SIZE) + 1]

    for model_item in model_list:

        print(f"Starting Speed Test for: {model_item.name}-{model_item.options}")

        model_index = len(df) # Store the index for updating while datasets run. 
        df.loc[model_index] = [model_item.name] + [model_item.options] + [0.0] * len(batch_size_subset)

        for batch_size in batch_size_subset:

            if model_item.FP16_precision == False and model_item.INT8_precision == False:
                precision_level = model_item.FP32_precision_level if model_item.FP32_precision_level else "highest"
                torch.set_float32_matmul_precision(precision_level)
                
            # Generate fuzzed data for the inputs
            input_shape = (batch_size, N_CHANNELS, IMG_SIZE, IMG_SIZE)

            if model_item.INT8_precision or device.type == 'cpu':
                cuda = False
            else:
                cuda = True
            
            fuzzer = benchmark.Fuzzer(
                    parameters = [],
                    tensors = [
                        benchmark.FuzzedTensor(
                            'x', 
                            size=input_shape, 
                            dtype=torch.half if model_item.FP16_precision else torch.float32, 
                            cuda=cuda,
                            probability_contiguous=0.6)            
                        
                    ],
                    seed=0,
                )
          
            for tensors, _, _ in fuzzer.take(1):
                input_data = tensors["x"]
                
            # Environment Setup
            vars = {
                'inference_test': inference_test, 
                'model_item': model_item, 
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

            print(f"{model_item.name}-{model_item.options} ({batch_size} Batch) Time: {benchmark_result.median * 1000} ms")
            print("")

            csv_path = f'{OUT_DIR}temp_speed_results.csv'
            df.to_csv(csv_path, index=False)

            gc.collect() # This may not be necessary, but due to some segmentation faults, this is here just in case for now. 
        
        print(f"Ending Speed Test for: {model_item.name}-{model_item.options}")

def run_accuracy_benchmark_tests(
        model_list: List[ModelInfo], 
        df: pd.DataFrame) -> None:
    
    dataset = datasets.ImageNet(DATA_DIR, split='val')

    # Use a different seed than what was used during 8 bit quantization.
    dataloader = get_dataloader_subset(dataset, CPU_MAX_BATCH_SIZE, seed=99) 

    for model_item in model_list:

        print(f"Starting Accuracy Test for: {model_item.name}-{model_item.options}")

        model_index = len(df) # Store the index for updating while datasets run. 
        df.loc[model_index] = [model_item.name] + [model_item.options] + [f'{CPU_MAX_BATCH_SIZE} Batch Size']
        
        model_item.model.eval()

        if model_item.FP32_precision_level is not None:
            precision_level = model_item.FP32_precision_level if model_item.FP32_precision_level else "highest"
            torch.set_float32_matmul_precision(precision_level)

        # Lists to store true and predicted labels
        true_preds = []
        all_preds = []
       
        for data, labels in dataloader:

            if model_item.FP16_precision:
                data = data.half()

            data = data.to(device)

            # Forward pass through the model
            outputs = model_item.model(data)
        
            # Compute predicted labels (e.g., argmax for classification)
            _, predicted = torch.max(outputs, 1)
            
            # Collect true and predicted labels
            true_preds.extend(labels)
            all_preds.extend(predicted.to('cpu').tolist())

        # Compute accuracy by comparing true and predicted labels
        accuracy = 100 * (np.array(true_preds) == np.array(all_preds)).mean()
        df.at[model_index, ACCURACY_COLUMN_NAME] = accuracy

        csv_path = f'{OUT_DIR}temp_accuracy_results.csv'
        df.to_csv(csv_path, index=False)

        print(f"{model_item.name}-{model_item.options} Accuracy: {accuracy}%")
        print(f"Ending Accuracy Test for: {model_item.name}-{model_item.options}")
        print("")

        gc.collect() # Using batch size 64 on the CPU, this is necessary to prevent memory issues. 

def inference_test(model_item: ModelInfo, input_data) -> None:
    with torch.no_grad():
        _ = model_item.model(input_data)


def CreateBarGraph(model_name: str, input_csv: str, output_png: str):

    # Read data from the CSV file
    df = pd.read_csv(input_csv)

    labels = df.iloc[:, 1]  

    # Get unique options from column 2
    unique_options = labels.unique()

    # Set up colors for each unique option
    colors = plt.cm.tab20.colors[:len(unique_options)]
    color_mapping = {option: color for option, color in zip(unique_options, colors)}

    # Get group names from the top of each column
    group_names = df.columns[2:9]

    # Create a figure and axis
    plt.figure(figsize=(12, 6))  # Set the figure size

    # Determine the bar width, group width, and positions
    group_count = len(group_names)
    x = np.arange(0, group_count)
    bars_per_group = len(unique_options)
    bar_width = 0.70 / bars_per_group
    

    # Create grouped bar graph with 3 bars for each unique option in each row and spaces between groups
   # for i, group_name in enumerate(group_names):
    for j, option in enumerate(unique_options):
        values = df[df.iloc[:, 1] == option].iloc[0, 2:9].values  # Get values for each option in the row
        
        x_offset = j * bar_width 
        plt.bar(
            x + x_offset,
            values,
            width=bar_width,
            label=f'{option}',
            color=color_mapping[option]
        )

    # Set x-axis ticks and labels
    plt.xticks(x, group_names)  # Set x-axis ticks based on group names
    plt.gca().set_xticks(x + (bars_per_group * bar_width) / 2.0 )  # Move the ticks to the left
    plt.gca().set_xticklabels(group_names, rotation=45)  # Set the tick labels with rotation

    # Set labels and title
    plt.xlabel('Batch Sizes')
    plt.ylabel('Time in ms')
    plt.title(f'{model_name} - Speed comparison')
    plt.suptitle('Speed in ms, based on batch size and compilation options', fontsize=12, fontweight='bold')

    # Create a legend in the upper left corner
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[option]) for option in unique_options]
    plt.legend(handles, unique_options, loc='upper left')

    # Show the plot
    plt.tight_layout()  # Adjust spacing for better appearance
    plt.savefig(output_png)


## Requires that the model be on the cuda device and set to eval mode
def trt_compile(model: nn.Module, is16Bit: bool = False):
    precision : {torch.dtype} = {torch_tensorrt.dtype.half} if is16Bit else {torch_tensorrt.dtype.float32}

    trt_module = torch_tensorrt.compile(
        model,
        inputs = [
            torch_tensorrt.Input(
                min_shape=[1, N_CHANNELS, IMG_SIZE, IMG_SIZE],
                opt_shape=[MAX_BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE],
                max_shape=[MAX_BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE],
            )],
        enabled_precisions = precision
    )

    return trt_module


# def eightbit_compile(model: nn.Module):

#     # Make sure we are both in eval mode, and 
#     # that make a copy of the model so that we don't
#     # modify the original.
#     orig_model = model.eval()
#     model = copy.deepcopy(orig_model)

#     # Convert the model to an FX graph
#     fx_model = torch.fx.symbolic_trace(model)

#     qconfig = get_default_qconfig("fbgemm")
#     qconfig_mapping = QConfigMapping().set_global(qconfig)

#     # Prepare the model for QAT using DataLoader to provide sample inputs
#     prepared_fx_model = prepare_fx(
#         fx_model,
#         qconfig_mapping,
#         get_sample_inputs())
    
#     prepared_fx_model = prepared_fx_model.to(device)
#     prepared_fx_model.eval()

#     quantized_model = convert_fx(prepared_fx_model)
#     quantized_model = quantized_model.to(device)
#     quantized_model.eval()
    
#     return quantized_model

def eightbit_compile(model: nn.Module):

    # Make sure we are both in eval mode, and 
    # that make a copy of the model so that we don't
    # modify the original.
    model.eval()

    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    # Prepare the model using DataLoader to provide sample inputs
    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, get_sample_inputs())
    
    # prepared_fx_model = prepared_fx_model.to(device)
    # prepared_fx_model.eval()

    quantized_model = convert_fx(prepared_model)
    quantized_model = quantized_model.to(device)
    quantized_model.eval()
    
    return quantized_model

def get_dataloader_subset(dataset: datasets.ImageNet, batch_size: int, seed: int = 42):

    # Select 20% of the dataset
    subset_fraction = 0.2  
    subset_size = int(len(dataset) * subset_fraction)

    # Create a subset of the larger dataset.
    torch.manual_seed(seed) 
    indices_shuffled = torch.randperm(len(dataset))
    subset_indices = indices_shuffled[:subset_size]
    subset_dataset = Subset(dataset, subset_indices)

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    dataloader = DataLoader(
        subset_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: (torch.stack([preprocess(item[0]) for item in batch]), [item[1] for item in batch]))
    
    return dataloader

def get_sample_inputs():

    # Using a different seed so that I get a different set of data to expose
    # the model to during quantization prep.
    dataset = datasets.ImageNet(DATA_DIR, split='val')
    dataloader = get_dataloader_subset(dataset, MAX_BATCH_SIZE, seed=99)

    for data, _  in dataloader:
        yield {'': data}


if __name__ == "__main__":
    main()
    