package com.example.filipino_sign_language;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Dialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.Toolbar;

import com.example.filipino_sign_language.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    androidx.appcompat.widget.Toolbar toolbar;
    Button btnPicture, btnGallery, btnClear;
    TextView txtDisplay;
    ImageView imgBox;
    int ImageSize = 384;
    Bitmap bitmap;
    Dialog dlInfo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        toolbar  = (androidx.appcompat.widget.Toolbar) findViewById(R.id.tlTop);
        btnPicture = findViewById(R.id.btnTakePicture);
        btnGallery = findViewById(R.id.btnOpenGallery);
        btnClear   = findViewById(R.id.btnClear);
        txtDisplay = findViewById(R.id.txtDisplay);
        imgBox     = findViewById(R.id.imgBox);
        dlInfo     = new Dialog(MainActivity.this);
        dlInfo.setContentView(R.layout.dialog_app_info);
        dlInfo.getWindow().setLayout(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        dlInfo.getWindow().setBackgroundDrawable(getDrawable(R.drawable.dl_box));
        dlInfo.setCancelable(true);

        setSupportActionBar(toolbar);

        btnPicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(cameraIntent, 1);
                    }else{
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, 100 );
                    }
                }
            }
        });

        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intentGallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intentGallery, 2);
            }
        });

        btnClear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                txtDisplay.setText(" ");
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.toolbar_top, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        int tlId = item.getItemId();
        if (tlId == R.id.tlInfo) {
            dlInfo.show();
        }
        return true;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode==RESULT_OK) {
            if (requestCode==1) {
                Bitmap photo = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(photo.getWidth(), photo.getHeight());
                photo = ThumbnailUtils.extractThumbnail(photo, dimension, dimension);
                imgBox.setImageBitmap(photo);
                photo = Bitmap.createScaledBitmap(photo, ImageSize, ImageSize, false);
                classifyImg(photo);

            } else if (requestCode == 2) {

                Uri dataUri = data.getData();
                Bitmap photo = null;
                try {
                    photo = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dataUri);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                int dimension = Math.min(photo.getWidth(), photo.getHeight());
                photo = ThumbnailUtils.extractThumbnail(photo, dimension, dimension);
                imgBox.setImageBitmap(photo);
                photo = Bitmap.createScaledBitmap(photo, ImageSize, ImageSize, false);
                classifyImg(photo);
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    public void classifyImg(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, ImageSize, ImageSize, 3}, DataType.FLOAT32);

            // ByteBuffer
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * ImageSize * ImageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Get Image Size from device
            // ImageSize is the same as the size for the data model
            int[] intValues = new int[ImageSize * ImageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < ImageSize; i++) {
                for (int j = 0; j <ImageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;

            for (int i = 0; i < confidences.length ; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"A", "B", "C", "D", "E", "F", "D", "E", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
            txtDisplay.append(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (Exception e) {
            System.out.println(e);
        }

    }
}
