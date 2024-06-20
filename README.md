# ImageTokenizer: Unified Image and Video Tokenization

Welcome to the **ImageTokenizer** repository! 🎉 This Python package is designed to simplify the process of image and video tokenization, a crucial step for various applications such as image/video generation and understanding. We provide a variety of popular tokenizers with a simple and unified interface, making your coding experience seamless and efficient. 🛠️

## Features

- **Unified Interface**: A consistent API for all supported tokenizers.
- **Extensive Support**: Covers a range of popular image and video tokenizers.
- **Easy Integration**: Quick setup and integration with your projects.

## Supported Tokenizers

Here's a list of the current supported image tokenizers:

- **OmniTokenizer**: Versatile tokenizer capable of handling both images and videos.
- **OpenMagvit2**: An open-source version of Magvit2, renowned for its excellent results.

## Getting Started

To get started with ImageTokenizer, follow these simple steps:

### Installation

You can install ImageTokenizer using pip:

```bash
pip install imagetokenizer
```

### Usage

Here's a quick example of how to use OmniTokenizer:

```python
from imagetokenizer import Magvit2Tokenizer

# Initialize the tokenizer
image_tokenizer = Magvit2Tokenizer()

# Tokenize an image
quants, embedding, codebook_indices = image_tokenizer.encode("path_to_your_image.jpg")

# Print the tokens
print(image_tokens)

image = image_tokenizer.decode(quants)
```

### Documentation

For more detailed information and examples, please refer to our [official documentation](#).

## Contributing

We welcome contributions! If you have an idea for a new tokenizer or want to improve existing ones, feel free to submit a pull request or create an issue. 🔧

## License

ImageTokenizer is open-source and available under the [MIT License](LICENSE).

## Community

- Join our [Slack Channel](#) to discuss and collaborate.
- Follow us on [Twitter](#) for updates and news.

## Acknowledgements

We would like to thank all the contributors and the community for their support and feedback. 🙏
