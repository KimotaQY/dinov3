def get_labels(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Please specify a dataset")

    if dataset_name == "Vaihingen" or dataset_name == "Potsdam":
        labels = [
            "roads", "buildings", "low veg.", "trees", "cars", "clutter"
        ]  # Label names
    elif dataset_name == "YYYJ":
        # LABELS = [
        #     "地基建设", "基础结构建设", "封顶厂房", "封顶楼房", "施工道路", "硬化道路", "风电施工", "风电", "光伏",
        #     "推填土", "体育场地", "临时棚房", "自建房", "未定义"
        # ]
        labels = [
            "地基建设", "基础结构建设", "封顶厂房", "封顶楼房", "施工道路", "硬化道路", "推填土", "体育场地",
            "临时棚房", "自建房", "专属设施", "未定义"
        ]

    return labels
