
import os
import sys
from pathlib import Path

from typing import TYPE_CHECKING

src_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')).resolve()

orig_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)

print(f'src path: {src_path}')
print(f'orig src path: {orig_src_path}')

if TYPE_CHECKING:
    # To make static type checker happy : )
    from src.Odometry.MACVO import MACVO
    from src.DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from src.Utility.Config import load_config
    from src.Utility.PrettyPrint import Logger
    from src.Utility.Timer import Timer
else:
    import DataLoader
    from Odometry.MACVO import MACVO                
    from DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from Utility.Config import load_config
    from Utility.PrettyPrint import Logger
    from Utility.Timer import Timer

print('Done importing')