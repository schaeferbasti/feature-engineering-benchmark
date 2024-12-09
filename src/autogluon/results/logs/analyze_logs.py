import glob
import os

import pandas as pd
from pandas import Index


def main():
    out_files = glob.glob('*.out')
    out_files = sorted(out_files)
    dataframe = pd.read_parquet("../evaldict.parquet")
    with open("analyze_logs.txt", 'w') as txt:
        txt.write("Evaluation of the Results \n\n")
    for out_file in out_files:
        err_file = out_file.replace(".out", ".err")
        with open(out_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(' binary-classification'):
                    method, dataset = get_method_and_dataset_name(line)
                    value = get_method_and_dataset_value(dataframe, method, dataset)
                    if value == "Failed" or value.startswith("0.0"):
                        with open("analyze_logs.txt", 'a') as txt:
                            txt.write(dataset + ", " + method + ", " + out_file + " (" + value + ") \n")
                    break
                if line.startswith(' regression'):
                    method, dataset = get_method_and_dataset_name(line)
                    value = get_method_and_dataset_value(dataframe, method, dataset)
                    if value == "Failed" or value.startswith("0.0"):
                        with open("analyze_logs.txt", 'a') as txt:
                            txt.write(dataset + ", " + method + " (" + value + ") \n")
                    if value == "None":
                        with open("analyze_logs.txt", 'a') as txt:
                            txt.write(dataset + ", " + method + " (No entry for this dataset) \n")
                    break
                if line.startswith(' multiclass'):
                    method, dataset = get_method_and_dataset_name(line)
                    value = get_method_and_dataset_value(dataframe, method, dataset)
                    if value == "Failed" or value.startswith("0.0"):
                        with open("analyze_logs.txt", 'a') as txt:
                            txt.write(dataset + ", " + method + " (" + value + ") \n")
                    break
        with open(err_file, 'r') as f:
            with open("analyze_logs.txt", 'r') as read_txt:
                err_lines = f.readlines()
                txt_lines = read_txt.readlines()
                if err_lines[-5] in txt_lines or err_lines[-4] in txt_lines or err_lines[-3] in txt_lines and err_lines[-2] in txt_lines or err_lines[-1] in txt_lines:
                    print("Skip")
                elif err_lines[-1].startswith("slurmstepd-kisexe"):
                    print("Skip")
                else:
                    with open("analyze_logs.txt", 'a') as append_txt:
                        append_txt.write(err_lines[-5])
                        append_txt.write(err_lines[-4])
                        append_txt.write(err_lines[-3])
                        append_txt.write(err_lines[-2])
                        append_txt.write(err_lines[-1] + "\n")
    with open("analyze_logs.txt", 'r') as f:
        txt_lines = f.readlines()
    with open("analyze_logs.txt", 'w') as f:
        for i in range(len(txt_lines) - 1):
            if not txt_lines[i].startswith(txt_lines[i+1][0:4]):
                f.write(txt_lines[i])


def get_method_and_dataset_name(line):
    splitted_line = line.split(' - ')
    method = splitted_line[-2].strip()
    dataset = splitted_line[-3].strip()
    return method, dataset


def get_method_and_dataset_value(dataframe, method, dataset):
    value_series = dataframe[method]
    datasets = dataframe["Dataset"]
    try:
        number = Index(datasets).get_loc(dataset)
        return value_series.values[number]
    except KeyError:
        return "None"


def rename_files():
    out_files = glob.glob('*.out')
    for out_file in out_files:
        with open(out_file, 'r') as file:
            lines = file.readlines()
        os.remove(out_file)
        split_out_file = out_file.split("_")
        split_out_file[-1] = split_out_file[-1].replace(".out", "")
        split_out_file[-1] = f"{int(split_out_file[-1]):04d}"
        split_out_file = "_".join(split_out_file)
        new_out_file = ".".join([split_out_file, "out"])
        print(new_out_file)
        with open(new_out_file, 'w') as file:
            file.writelines(lines)
    err_files = glob.glob('*.err')
    for err_file in err_files:
        with open(err_file, 'r') as file:
            lines = file.readlines()
        os.remove(err_file)
        split_err_file = err_file.split("_")
        split_err_file[-1] = split_err_file[-1].replace(".err", "")
        split_err_file[-1] = f"{int(split_err_file[-1]):04d}"
        split_err_file = "_".join(split_err_file)
        new_err_file = ".".join([split_err_file, "err"])
        print(new_err_file)
        with open(new_err_file, 'w') as file:
            file.writelines(lines)


if __name__ == '__main__':
    # rename_files()
    main()
