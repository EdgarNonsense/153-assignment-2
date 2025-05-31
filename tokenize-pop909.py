import os
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob

from tqdm import tqdm

# These are assumed to be defined in your anticipation.config module
# e.g., DELTA, MAX_TRACK_TIME_IN_SECONDS, MIN_TRACK_TIME_IN_SECONDS,
# MIN_TRACK_EVENTS, PREPROC_WORKERS, M (sequence length for stats)
from anticipation.config import *
from anticipation.tokenize import tokenize, tokenize_ia

def main(args):
    encoding = 'interarrival' if args.interarrival else 'arrival'
    print(f'Tokenizing preprocessed MIDI files from: {args.datadir}')
    print(f'  Encoding type: {encoding}')

    print('Tokenization parameters:')
    # Assuming DELTA, MAX_TRACK_TIME_IN_SECONDS, etc. are available from anticipation.config
    print(f'  Anticipation interval = {DELTA}s')
    print(f'  Augment = {args.augment}x')
    print(f'  Max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  Min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  Min track events = {MIN_TRACK_EVENTS}')

    # Get all .mid.compound.txt files from the specified directory
    input_files = sorted(glob(os.path.join(args.datadir, '*.mid.compound.txt')))

    if not input_files:
        print(f"No '*.mid.compound.txt' files found in {args.datadir}")
        return

    print(f'Found {len(input_files)} files to process.')

    # Define a single output file path (e.g., in the same directory as the input files)
    output_file = os.path.join(args.datadir, 'tokenized-events-all.txt')

    # Augmentation factor for the dataset
    current_augment_factor = args.augment

    func = tokenize_ia if args.interarrival else tokenize

    # Prepare arguments for starmap.
    # The tokenize function expects a list of input files, an output file path,
    # an augmentation factor, and a process index (for tqdm positioning).
    # We are treating all files as a single "job" or "split".
    starmap_args = [
        (input_files, output_file, current_augment_factor, 0) # proc_idx = 0 for the single job
    ]

    # PREPROC_WORKERS is assumed to be from anticipation.config
    # RLock is for tqdm to work correctly with multiprocessing
    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        # pool.starmap will call func with the arguments provided in starmap_args.
        # Since starmap_args contains one tuple, func will be called once.
        results = pool.starmap(func, starmap_args)

    if not results:
        print("Tokenization did not return any results. This might indicate an issue.")
        return

    # results will be a list containing one tuple of statistics from the tokenize function
    # e.g., [(seq_count, rest_count, too_short, ...)]
    seq_count, rest_count, too_short, too_long, too_manyinstr, discarded_seqs, truncations = results[0]

    # M is sequence length, assumed to be from anticipation.config for calculating ratios
    rest_ratio = 0
    trunc_ratio = 0
    if seq_count > 0 and M > 0: # Avoid division by zero
        rest_ratio = round(100 * float(rest_count) / (seq_count * M), 2)
        trunc_type = 'interarrival' if args.interarrival else 'duration'
        trunc_ratio = round(100 * float(truncations) / (seq_count * M), 2)
    else:
        trunc_type = 'interarrival' if args.interarrival else 'duration'


    print('Tokenization complete.')
    print(f'  => Output file: {output_file}')
    print(f'  => Processed {seq_count} sequences')
    print(f'  => Inserted {rest_count} REST tokens ({rest_ratio}% of events)')
    print(f'  => Discarded {too_short + too_long} event sequences')
    print(f'      - {too_short} too short (less than {MIN_TRACK_EVENTS} events or {MIN_TRACK_TIME_IN_SECONDS}s)')
    print(f'      - {too_long} too long (more than {MAX_TRACK_TIME_IN_SECONDS}s)')
    print(f'      - {too_manyinstr} had too many instruments') # Assuming this stat comes from tokenize function
    print(f'  => Discarded {discarded_seqs} sequences (e.g. empty after filtering)')
    print(f'  => Truncated {truncations} {trunc_type} times ({trunc_ratio}% of {trunc_type}s)')

    print('If this tokenized data is intended for training, remember to shuffle it afterwards.')

if __name__ == '__main__':
    parser = ArgumentParser(description='Tokenizes a MIDI dataset represented by .mid.compound.txt files.')
    parser.add_argument('datadir', help='Directory containing preprocessed MIDI files (e.g., *.mid.compound.txt)')
    parser.add_argument('-k', '--augment', type=int, default=1,
            help='Dataset augmentation factor (default: 1, no augmentation)')
    parser.add_argument('-i', '--interarrival',
            action='store_true',
            help='Use interarrival-time encoding (defaults to arrival-time encoding)')

    main(parser.parse_args())