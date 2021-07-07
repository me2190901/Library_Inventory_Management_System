import spine_segment
import glob as gb


img_path = gb.glob("./images/\\*.jpg")
for path in img_path:
    spine_segment.get_book_lines(path)
# spine_segment.get_book_lines("./images/1.jpg")
print("Segmentation_Done")
