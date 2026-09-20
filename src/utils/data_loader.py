from datasets import load_dataset

def load_tofu_dataset(split="train", max_samples=50):
    print(f"Loading TOFU dataset...")
    dataset = load_dataset("locuslab/TOFU", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    samples = []
    for idx, item in enumerate(dataset):
        samples.append({
            'sample_id': idx,
            'text': item['question'] + " " + item['answer'],
            'is_member': True
        })
    
    return samples

def load_custom_dataset(filepath, text_column="text", label_column="is_member"):
    """
    Load custom CSV/JSON dataset.
    
    Args:
        filepath: Path to CSV or JSON file
        text_column: Name of text column
        label_column: Name of membership label column (optional)
    
    Returns:
        List of dicts
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        raise ValueError("File must be .csv or .json")
    
    samples = []
    for idx, row in df.iterrows():
        samples.append({
            'sample_id': idx,
            'text': row[text_column],
            'is_member': row.get(label_column, None)
        })
    
    return samples