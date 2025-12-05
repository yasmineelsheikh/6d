def get_command(params: dict, output_path: str) -> str:
    command = f"python cosmos-transfer2.5/examples/inference.py -i {params} -o {output_path}"
    return command