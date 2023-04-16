import argparse
from numpy import arange
from tqdm import tqdm

from Code.models import CFN, CFN_transfer, ResNet34_refrence
from Code.transfer_learning import perform_transfered_task

# function that determines the function call parameters and returns them
def get_arguments():
    parser = argparse.ArgumentParser(description='Deep Learning Project')

    # arguments allowed in function call
    parser.add_argument('--task', type=str, default="class", help='What task to be run, semi-supervised learning (semi), self-supervised (self) or simple classification (class)')
    # Model type can be either cfn_transfered or resnet for simple clasification or it will default to cfn for self-supervised and cfn_transfered for semi-supervised
    parser.add_argument('--model', type=str, default="resnet", help='What type of model to be used for simple classification (cfn_transfered, resnet)')
    
    parser.add_argument('--dataset', type=str, default="cifar10", help='What dataset to be used (cifar10 or stl10)')
    
    # Batch size of 256 used for training (not advised, very computational intensive)
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size')
    
    arguments = parser.parse_args()
    # raising errors in case more than two values (a range) are given for the learning rate, batch size 
    if arguments.task not in ["semi", "self", "class"]:
        raise parser.error("Mentioned task not existant, choose either 'semi', 'self' or 'class' for --task")
    
    if arguments.model not in ["cfn_transfered", "resnet"]:
        raise parser.error("Model for simple classification should be either 'cfn_transfered' or 'resnet' for --model")
    
    if arguments.dataset not in ["cifar10", "stl10"]:
        raise parser.error("The dataset should be either 'cifar10' or 'stl10' for --dataset")
    
    if arguments.task == "semi":
        print("Warning: Model defaults to cfn_transfered for semi-supervised task")

    if arguments.task == "self":
        print("Warning: Model defaults to cfn for self-supervised task")

    # returning each function call parameter
    return arguments


def run(task, model_type, dataset="cifar10", batch_size=64):
    if dataset == "cifar10":
        image_dimensions = 32
        crop_dimensions = 7
    else:
        image_dimensions = 96
        crop_dimensions = 27
    
    if task == "class":
        if model_type == "resnet":
            model = ResNet34_refrence(dataset=dataset, image_dimensions=image_dimensions)
        elif model_type == "cfn_transfered":
            model = CFN_transfer(dataset=dataset, image_dimensions=image_dimensions)
        
        model.summary()
        model.train(batch_size=batch_size)
        model.test()
    
    elif task == "self":
        model = CFN(dataset=dataset, crop_dimensions=crop_dimensions)

        model.summary()
        model.train(batch_size=batch_size)
        model.test()
    
    elif task == "semi":
        perform_transfered_task(dataset=dataset, batch_size=batch_size)


if __name__ == '__main__':
    arguments = get_arguments()
    
    # To not perform a part of the task mention it in run command line. For example: -no-training
    run(
        task=arguments.task,
        model_type=arguments.model,
        dataset=arguments.dataset,
        batch_size=arguments.batch_size
        )