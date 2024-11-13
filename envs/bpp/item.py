from typing import Dict, Union, Optional

__all__ = ["Item"]


class Item(object):
    def __init__(
        self,
        width: int, 
        height: int, 
        lx: int, 
        ly: int, 
        isplaced: bool = None
    ):
        """
        Record the item information placed in the bin.

        Parameters
        ----------
        width: int
            The width of the item.
        height: int
            The height of the item.
        lx: int
            The x-coordinate of the width of the item.
        ly: int
            The y-coordinate of the height of the item.
        isplaced: bool (default is None)
            Whether the item is placed in the bin.

        Examples
        --------
        >>> item = Item(1, 2, 0, 0)
        >>> print(repr(item))
        Item(width=1, height=2, lx=0, ly=0, isplaced=None)
        """
        self._width = width        # width of item
        self._height = height      # height of item
        self._lx = lx              # location x of width of item
        self._ly = ly              # location y of height of item
        self._isplaced = isplaced  # whether the item is placed
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    @property
    def lx(self) -> int:
        return self._lx
    
    @property
    def ly(self) -> int:
        return self._ly
    
    @property
    def isplaced(self) -> bool:
        return self._isplaced

    @isplaced.setter
    def isplaced(self, value: bool):
        if isinstance(value, bool):
            self._isplaced = value
        else:
            raise ValueError("is_placed must be a boolean value")
        
    @property
    def size(self) -> Dict[str, int]:
        return dict(
            width=self._width,
            height=self._height
        )
    
    @property
    def location(self) -> Dict[str, int]:
        return dict(
            lx=self._lx,
            ly=self._ly
        )

    @property
    def info(self) -> Dict[str, Optional[Union[int, bool]]]:
        return dict(
            width=self._width,
            height=self._height,
            lx=self._lx,
            ly=self._ly,
            isplaced=self._isplaced
        )

    def __str__(self) -> str:
        if self.isplaced:
            return f"item with size {self.size} is placed at location {self.location}"
        else: 
            return f"item with size {self.size} is not placed"

    def __repr__(self) -> str:
        return f"Item(width={self.width}, height={self.height}, lx={self.lx}, ly={self.ly}, isplaced={self.isplaced})"