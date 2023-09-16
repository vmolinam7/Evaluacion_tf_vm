package com.example.evaluacion_tf_vm;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.evaluacion_tf_vm.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_REQUEST_CODE = 100;
    private static final int GALLERY_REQUEST_CODE = 200;

    private ImageView imageView;
    private Bitmap inputImage;
    private Interpreter tflite;

    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        resultTextView=findViewById(R.id.resultTextView);
        imageView = findViewById(R.id.img_persona);
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void openCamera(View view) {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, CAMERA_REQUEST_CODE);
        }
    }

    public void openGallery(View view) {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, GALLERY_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == CAMERA_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            Bundle extras = data.getExtras();
            if (extras != null) {
                inputImage = (Bitmap) extras.get("data");
                imageView.setImageBitmap(inputImage);
                processesImage(inputImage);
            }
        } else if (requestCode == GALLERY_REQUEST_CODE && resultCode == Activity.RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            try {
                inputImage = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                imageView.setImageBitmap(inputImage);
                processesImage(inputImage);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private MappedByteBuffer loadModelFile() throws Exception {
        FileInputStream inputStream = new FileInputStream("ModelUnquant.tflite");
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = 0;
        long declaredLength = fileChannel.size();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void processImageWithTensorFlow(Bitmap bitmap) {
        if (tflite != null) {
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 350, 350, true);
            int[] intValues = new int[350 * 350];
            resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());
            float[][] inputArray = preprocessImage(intValues);
            float[][] outputArray = new float[1][1];
            tflite.run(inputArray, outputArray);
            float predictedValue = outputArray[0][0];
            int predictedClass = predictedValue >= 0.5f ? 1 : 0;
            imageView.setImageBitmap(bitmap);
        }
    }


    private float[][] preprocessImage(int[] intValues) {
        int inputWidth = 350;
        int inputHeight = 350;
        float[][] inputArray = new float[1][inputWidth * inputHeight * 3];

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            float normalizedValue = ((val & 0xFF) / 255.0f);
            inputArray[0][i * 3 + 0] = normalizedValue; // R
            inputArray[0][i * 3 + 1] = normalizedValue; // G
            inputArray[0][i * 3 + 2] = normalizedValue; // B
        }

        return inputArray;
    }

    public void processesImage(Bitmap image){
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int [] intValues = new int[224 * 224];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for(int i = 0; i < 224; i++){
                for(int j = 0; j < 224; j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] resul = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxPro = 0;
            for(int i = 0; i < resul.length; i++){
                if(resul[i] > maxPro){
                    maxPro = resul[i];
                    maxPos = i;
                }
            }
            String[] personas = {"Victor", "El Hombre Araña"};
            resultTextView.setText(personas[maxPos]);
            String s = "La imagen pertenece a:";
            for(int i = 0; i < personas.length; i++){
                s += String.format("%s: %.1f%%\n", personas[i], resul[i] * 100);
            }
            model.close();
        } catch (IOException e) {
            Log.i("error", e.toString());
        }
    }

    @Override
    protected void onDestroy() {
        if (tflite != null) {
            tflite.close();
        }
        super.onDestroy();
    }
}


