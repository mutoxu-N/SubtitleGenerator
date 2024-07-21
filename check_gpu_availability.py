if __name__ == "__main__":
    import torch

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"{count} GPUs detected.")

        device = torch.cuda.current_device()
        print(f"Default GPU: {device}, name: {
              torch.cuda.get_device_name(device)}")

    else:
        print("No GPUs detected.")
