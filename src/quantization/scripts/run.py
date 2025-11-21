from .utils import predict_file

import typer
import torch


app = typer.Typer()


@app.command()
def generate_from_ptq_int8(
    weights_int8_path: str = typer.Option(...),
    input: str = typer.Option(),
    output: str = typer.Option(),
    batch_size: int = typer.Option()
):
    from PTQ.generator import Generator

    model = Generator().eval()
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    model = torch.quantization.prepare(model)
    model = torch.quantization.convert(model)

    state = torch.load(weights_int8_path, map_location=torch.device('cpu'))
    model.load_state_dict(state)

    predict_file(model, input, output, batch_size)


if __name__ == "__main__":
    app()