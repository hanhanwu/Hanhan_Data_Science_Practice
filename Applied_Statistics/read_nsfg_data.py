"""
Read NSFG data and convert to csv files for later analysis
"""
import csv


def read_NSFGfile(file_path, col_name_dct):
    """
    Read NSFG file
    Get each col value based on (start, end) index
    :param file_path: input file path
    :param col_name_dct: The dictionary that contains colname as well as the index range
    :return: extracted data
    """
    results = []
    with open(file_path) as filein:
        for r in filein:
            tmp_dct = {}
            for col_name, col_range in col_name_dct.items():
                val = r[col_range[0]:col_range[1]]
                if val is None or val.split()==[]:
                    tmp_dct[col_name] = None
                else:
                    if col_name == "finalwgt":
                        tmp_dct[col_name] = float(r[col_range[0]:col_range[1]])
                    else:
                        tmp_dct[col_name] = int(r[col_range[0]:col_range[1]])

            results.append(tmp_dct)
    return results


def write_csv(data_lst, output_file, cols):
    """
    Write extracted data into csv file
    :param data_lst: extracted data
    :param output_file: output csv file path
    :param cols: colname list
    :return: None
    """
    with open(output_file, 'w') as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=cols)
        writer.writeheader()

        for r in data_lst:
            writer.writerow(r)


def main():
    preg_file = "2002FemPreg.csv"  # not csv file....
    preg_col_name_dct = {"caseid":(0, 12), "nbrnaliv":(21, 22), "babysex":(55, 56), "birthwgt_lb":(56, 58),
                         "birthwgt_oz":(58,60), "prglngth":(274,276), "outcome":(276, 277), "birthord":(277, 279),
                         "agepreg":(283,287), "finalwgt":(422, 440)}
    preg_results = read_NSFGfile(preg_file, preg_col_name_dct)
    print(len(preg_results))
    preg_col_names = ["caseid", "nbrnaliv", "babysex", "birthwgt_lb", "birthwgt_oz", "prglngth", "outcome", "birthord",
                      "agepreg", "finalwgt"]
    output_file = "2002FemPregOut.csv"
    write_csv(preg_results, output_file, preg_col_names)


    resp_file = "2002FemResp.csv"   # not csv file....
    resp_col_name_dct = {"caseid":(0, 12)}
    resp_results = read_NSFGfile(resp_file, resp_col_name_dct)
    print(len(resp_results))
    resp_col_names = ["caseid"]
    output_file = "2002FemRespOut.csv"
    write_csv(resp_results, output_file, resp_col_names)

if __name__ == "__main__":
    main()
