def visualiseDetectron():
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

  # Prikaz slike sa predikcijama
  plt.figure(figsize=(12, 12))
  plt.imshow(out.get_image()[:, :, ::-1])
  plt.axis('off')
  plt.show()
  