import tensorflow as tf

# Đọc bộ não cũ
model = tf.keras.models.load_model('forecast_model.h5', compile=False)

# Chuyển sinh sang định dạng chuẩn mới nhất
model.save('forecast_model.keras')