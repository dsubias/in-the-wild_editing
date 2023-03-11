import os

folder = "video_headpod"
out = open("data_45k/" + folder + ".txt","w")
files = os.listdir("data_45k/renderings/" + folder)
files.sort()
out.write("plastic	rubber	metallic	glossy	bright	rough	refstrng	refsharp	\n")

for file in files:
    out.write(folder + "/" + file + " 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")

out.close()