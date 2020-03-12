import os


def export_layers(img, drw, path, name):
    img = img.duplicate()

    for layer in img.layers:
        layer.visible = False

    for idx, layer in enumerate(img.layers):
        layer.visible = True

        filename = name % [idx, layer.name]
        fullpath = os.path.join(path, filename)

        layer_img = img.duplicate()
        # layer_img.flatten()

        # pdb.gimp_file_save(layer_img, drw, fullpath, filename)
        pdb.file_png_save(
            img, layer, fullpath + filename, "", False, 9, True, True, True, True, True
        )
        # img.remove_layer(layer)


def export_all_layers(img, path):
    for layer in img.layers:
        export_layers(img, layer, path, layer.name + ".png")


os.chdir("/Users/andriylevitskyy/Desktop")
export_all_layers(gimp.image_list()[0], "/Users/andriylevitskyy/Desktop/")
