def phase1_check():
    print("=" * 50)
    print("LLM Lab — Phase 1 check")
    print("=" * 50)

    print("\n[1/2] Config system...")
    from configs.config_loader import load_config
    configs = load_config()
    print(f"  OK — {len(configs)} experiments loaded")
    for c in configs:
        print(f"       {c.name}")

    print("\n[2/2] Dataset module...")
    from data.loader import load_samples
    samples = load_samples("alpaca", num_samples=5)
    print(f"  OK — {len(samples)} samples loaded")
    print(f"  Sample length: {len(samples[0]['text'])} chars")

    print("\n" + "=" * 50)
    print("Phase 1 complete. Ready for Phase 2.")
    print("=" * 50)

if __name__ == "__main__":
    phase1_check()