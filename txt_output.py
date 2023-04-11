def yolo_txt_maker(index, xmin, ymin, xmax, ymax, x, y):
    """
    生成yolo格式的txt文件
    :param index: 类别序号
    :param x: 图片宽度
    :param y: 图片长度
    :return: txt文件内容
    """
    yolo_data = "{index} {position_x} {position_y} {width} {height}".format(
        index=str(index), position_x=str((xmin + xmax) / 2 / x), position_y=str((ymin + ymax) / 2 / y)
        , width=str((xmax - xmin) / x), height=str((ymax - ymin) / x))
    return yolo_data


def voc_xml_maker(file_name, xmin, ymin, xmax, ymax, kind, picture_width, picture_height):
    """
    生成voc格式的xml文件
    :param file_name: 图片文件名
    :param kind: 物品种类
    :return: xml文本字符串
    """
    voc_data = '''<?xml version="1.0" ?>
    <annotation>
    <folder>something</folder>
    <filename>{file_name}</filename>
    <path>{file_name}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{picture_width}</width>
        <height>{picture_height}</height>
        <depth>3</depth>
    </size>

    <segmented>0</segmented>
        <object>
        <name>{kind}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
    </annotation>
    '''.format(file_name=file_name, xmin=str(xmin), ymin=str(ymin), xmax=str(xmax), ymax=str(ymax), kind=kind
               , picture_width=str(picture_width), picture_height=str(picture_height))
    return voc_data
