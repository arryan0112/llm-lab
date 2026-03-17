from data.loader import load_samples

def test_all():
    for name in ["alpaca", "squad", "cnn"]:
        print(f"\nTesting {name}...")
        samples = load_samples(name, num_samples=3)
        print(f"  Got {len(samples)} samples")
        print(f"  Preview:\n  {samples[0]['text'][:200]}")
        print(f"  {'─'*50}")

if __name__ == "__main__":
    test_all()