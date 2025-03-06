from typing import List, Tuple

import copy
import math
import numpy as np

__all__ = ["ItemsCreator"]


class ItemsCreator(object):
    def __init__(
        self, 
        bin_w: int = 10, 
        bin_h: int = 10, 
        min_items_per_bin: int = 10,
        max_items_per_bin: int = 20,
    ):
        """
        A item creator object to generate items list for the BPP environment.\n
        The item creator object will generate a list of items for each episode.

        Parameters
        ----------
        bin_w: int
            The width of the bin.
        bin_h: int
            The height of the bin.
        items_per_bin: int
            The number of items per bin.
        """
        self.bin_w = bin_w
        self.bin_h = bin_h
        self.min_items_per_bin = min_items_per_bin
        self.max_items_per_bin = max_items_per_bin
        self.items_per_bin = None
        self.items_list: List[Tuple[int, int]] = []

    def reset(
        self,
        np_random: np.random.Generator
    ) -> None:
        """
        Clear the current `items_list`, and then generate a new item datas for the new episode.

        Parameters
        ----------
        np_random: np.random.Generator
            A instances of `np.random.Generator` passed from the `env.np_random`.
        """
        assert np_random is not None, "The `np_random` should not be ."
        self.items_list.clear()
        self.items_per_bin = np_random.integers(self.min_items_per_bin, self.max_items_per_bin + 1)
        self.items_list = self._generate_items_list(np_random=np_random)

    def preview(self, length: int = 1) -> List[Tuple[int, int]]:
        """
        Return the coming items according to the preview length from the `items_list`.

        Parameters
        ----------
        length : int (default=1)
            The number of items to preview. 
        
        Returns
        -------
        out: List[Tuple[int, int]]
            The coming items list. 

        Examples
        --------
        >>> items_creator = ItemsCreator(bin_w=5, bin_h=5)
        >>> items_creator.reset()
        >>> items_creator.preview(length=2)
        [(2, 3), (4, 1)]
        """
        while (self.items_list_length() < length):
            self._add_unit_item()
        return copy.deepcopy(self.items_list[:length])

    def _add_unit_item(self) -> None:
        """
        Add a unit item (1,1) in the list, to ensure all next state is able to be observed.

        Notes
        -------
        Only used in the `self.preview()` function.
        """
        self.items_list.append((1,1))

    def drop_item(self) -> None:
        """
        Remove the first item from the `items_list`.

        Notes
        -------
        Only used in the gym.step function.
        """
        assert (self.items_list_length() > 0), "The current `items_list` is empty."
        self.items_list.pop(0)

    def items_list_length(self) -> int:
        """
        Return the length of the `items_list`.

        Returns
        -------
        out: int
            The length of the `items_list`.
        """
        return len(self.items_list)
    
    def _generate_items_list(
        self, 
        np_random: np.random.Generator,
    ) -> List[Tuple[int, int]]:
        """
        Generate a new items list.
        
        Parameters
        ----------
        np_random: np.random.Generator
            A instances of `np.random.Generator` passed from the `env.np_random`.
        
        Notes
        -------
        Only used in the `self.reset()` function.
        """
        # check size and items number per bin
        if self.items_per_bin > (self.bin_w * self.bin_h):
            raise ValueError(f"Bin doesn't have enough place to cut, max items number is: {int(self.bin_w * self.bin_h)}")
        
        ''' width '''
        # 先決定在 width 方向要切成幾塊，至少要切 1 塊(即不切)，至多切 min(items_per_bin, bin_w) 塊
        width_cut_number = np_random.integers(
            math.ceil(self.items_per_bin / self.bin_h), min(self.items_per_bin, self.bin_w)+1)

        # 決定切點位置，再加上頭尾兩個切點位置
        width_cut_point = np_random.choice(range(1, self.bin_w), (width_cut_number-1), replace=False).tolist()
        width_cut_point.insert(0, 0)
        width_cut_point.append(self.bin_w)
        # print(width_cut_point)
        # print(type(width_cut_point))
        width_cut_point.sort()

        # 計算相鄰切點之間的距離，即每塊的寬度
        width_cut_data = []
        for i in range(width_cut_number):
            width_cut_data.append((width_cut_point[i+1] - width_cut_point[i]))

        ''' height '''
        # 決定 width 方向的每一區段的物品數量，初始化每一區段都切 1 片
        height_cut_numbers = [1] * width_cut_number
        total_items = width_cut_number
        while (total_items != self.items_per_bin):
            choice = np_random.integers(0, width_cut_number)
            if (height_cut_numbers[choice] < self.bin_h):
                height_cut_numbers[choice] += 1
                total_items += 1

        # 決定 width 方向每一塊的切點位置
        bin_cut_data = []
        for index, height_cut_number in enumerate(height_cut_numbers):
            # 決定切點位置，再加上頭尾兩個切點位置
            height_cut_point = np_random.choice(range(1, self.bin_h), (height_cut_number-1), replace=False).tolist()
            height_cut_point.insert(0, 0)
            height_cut_point.append(self.bin_h)
            height_cut_point.sort()
            # 計算相鄰切點之間的距離，即每塊的長度
            height_cut_data = []
            for i in range(height_cut_number):
                height_cut_data.append((height_cut_point[i+1] - height_cut_point[i]))
            # 儲存每塊 item 的寬高
            for i in range(len(height_cut_data)):
                bin_cut_data.append((width_cut_data[index], height_cut_data[i]))
        bin_cut_data_no_shuffle = copy.deepcopy(bin_cut_data)
        np_random.shuffle(bin_cut_data)
        bin_cut_data_shuffle = copy.deepcopy(bin_cut_data)
        return bin_cut_data_shuffle
