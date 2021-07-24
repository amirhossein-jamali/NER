

def get_data(file_path):
    """Read sentences and tags from a data file.

    Args:
      file_path (str): path to NER data file.

    Returns:
      examples (dict): a dictionary with two keys: words (list of lists)
        holding words in each sequence, and labels (list of lists) holding
        corresponding labels.
    """
    with open(file_path, encoding="utf-8") as f:
        examples = {"sentences": [], "tags": []}
        sentences = []
        tags = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if sentences:
                    examples["sentences"].append(sentences)
                    examples["tags"].append(tags)
                    sentences = []
                    tags = []
            else:
                splits = line.split(" ")
                sentences.append(splits[0])
                if len(splits) > 1:
                    tags.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    tags.append("O")
    return examples
