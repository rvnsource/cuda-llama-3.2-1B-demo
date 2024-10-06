import torch

print("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    print('CUDA available')

    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print("Devices available in this machine:")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))

else:
    print('CUDA not available')

