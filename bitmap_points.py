import numpy as np
from PIL import Image
from os.path import exists


class BitMapPoints():
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        if exists(self.file_path):
            try:
                self.image = Image.open(self.file_path).convert('L')
            except Exception as exc:
                print(f'Error occurred while loadin image. Details below:\n\t{exc}')
        else:
            print('File do not exists.')

    def convert_to_list_of_points(self) -> list[tuple]:
        image_points = np.array(self.image)
        tuple_list = []
        for row in range(len(image_points)):
            for column in range(len(image_points[row])):
                if image_points[row][column] <= 200:
                    tuple_list.append((row, column))
        return tuple_list
