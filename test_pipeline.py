import sys
sys.path.insert(0, ".")
from inference.pipeline import InferencePipeline

def test_basic_generation():
    pipe = InferencePipeline()
    img = pipe.generate(
        prompt="a cute cat in a forest",
        num_inference_steps=20,
        guidance_scale=7.5,
    )
    img.save("test_output.png")
    print("Image saved to test_output.png")

if __name__ == "__main__":
    test_basic_generation()
