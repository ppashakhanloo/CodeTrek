import sys
import hashlib
from os import listdir
from os.path import isfile, join
from typing import List, Dict


class OverlapChecker:

    train_dir = None  # type: str
    dev_dir = None    # type: str
    test_dir = None   # type: str

    def __init__(self, train: str, dev: str, test: str):
        self.train_dir = train
        self.dev_dir = dev
        self.test_dir = test

    @staticmethod
    def build_digest_map(directory: str) -> Dict[str, str]:
        digest_file_map = {}
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        for f in files:
            with open(join(directory, f), 'r') as file:
                data = file.read()
                data = "".join(data.split())
                digest = hashlib.sha256(data.encode('utf-8')).hexdigest()
                if digest in digest_file_map.keys():
                    print(f'Overlap warning: {f} == {digest_file_map[digest]}')
                digest_file_map[digest] = f
        return digest_file_map

    @staticmethod
    def check_two(map1: Dict[str, str], map2: Dict[str, str], name1: str, name2: str) -> None:
        intersection = map1.keys() & map2.keys()
        if len(intersection) == 0:
            print(f'No overlap between {name1} and {name2}')
        else:
            for digest in intersection:
                print(f'Overlap warning: {name1}:{map1[digest]} and {name2}:{map2[digest]}')

    def check(self):
        train_map = OverlapChecker.build_digest_map(self.train_dir)
        dev_map = OverlapChecker.build_digest_map(self.dev_dir)
        test_map = OverlapChecker.build_digest_map(self.test_dir)
        OverlapChecker.check_two(train_map, dev_map, 'train', 'dev')
        OverlapChecker.check_two(train_map, test_map, 'train', 'test')
        OverlapChecker.check_two(dev_map, test_map, 'dev', 'test')


def main(args: List[str]):
    if not len(args) == 4:
        print('Usage: python3 check_overlap.py <train-dir> <dev-dir> <test-dir>')
        exit(1)

    train_dir = args[1]
    dev_dir = args[2]
    test_dir = args[3]

    checker = OverlapChecker(train_dir, dev_dir, test_dir)
    checker.check()


if __name__ == '__main__':
    main(sys.argv)