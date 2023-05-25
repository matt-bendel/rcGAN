import os
import yaml
import json
import types
import shutil


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    with open('configs/mri.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    with open('scripts/mri/mr_test.txt') as f:
        test_files = f.read().split('\n')

    os.mkdir(os.path.join(cfg.data_path, 'small_T2_test'))

    for file in test_files:
        old_file_path = os.path.join(cfg.data_path, f'multicoil_val/{file}')
        new_file_path = os.path.join(cfg.data_path, f'small_T2_test/{file}')

        shutil.move(old_file_path, new_file_path)