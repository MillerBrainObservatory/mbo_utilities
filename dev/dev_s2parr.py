from mbo_utilities.lazy_array import imread
import fastplotlib as fpl

data = imread(r"D:\W2_DATA\kbarber\07_27_2025\mk355\zarr\s2p-fix-reg-force-refimg")
iw = data.imshow()
iw.show()
fpl.loop.run()

print(data.shape)