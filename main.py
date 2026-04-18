import pandas as pd
from datasets import Dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input JSON')
    parser.add_argument('--output', required=True, help='Path to output JSON')
    args = parser.parse_args()
    
    pass

if __name__ == '__main__':
    main()