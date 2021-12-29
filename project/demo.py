"""Demo."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 17日 星期四 14:09:56 CST
# ***
# ************************************************************************************/
#

import face_motion

if __name__ == "__main__":
    face_motion.predict("videos/2.mp4", "images/feynman.jpeg", "output/face_predict.mp4")
    face_motion.client("PAI", "videos/2.mp4", "images/feynman.jpeg", "output/face_server.mp4")
    face_motion.server("PAI")

