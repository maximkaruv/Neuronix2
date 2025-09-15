import os

def print_tree(start_path=".", prefix=""):
    items = sorted(os.listdir(start_path))
    for index, item in enumerate(items):
        if item == "__pycache__" or item == ".git":
            continue  # пропускаем __pycache__
        path = os.path.join(start_path, item)
        connector = "├── " if index < len(items) - 1 else "└── "
        print(prefix + connector + item)
        if os.path.isdir(path):
            extension = "│   " if index < len(items) - 1 else "    "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print(".")
    print_tree(".")
