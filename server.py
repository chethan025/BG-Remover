from backgroundremover.bg import remove

remove(
    src_img_path="input.jpg",
    out_img_path="output.png",
    model_choice="u2net"
)
print("Background removal completed. Check output.png for the result.")