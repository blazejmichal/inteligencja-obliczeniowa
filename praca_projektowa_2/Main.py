from praca_projektowa_2.DataFrameLoader import DataFrameLoader


def main():
    print("Praca projektowa 2")
    data_frame_loader = DataFrameLoader()
    # data_frame_loader = DataFrameLoader.build(1, 'test', 'test')
    data_frame = data_frame_loader.run()
    print(data_frame)


if __name__ == '__main__':
    main()
