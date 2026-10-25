# Reference Chest X-ray Images Repository

This directory contains the reference Chest X-ray images used for similarity comparison.

## Adding Reference Images
1. **Place your permitted/reference Chest X-ray images inside `reference_images/`**.
2. Supported formats: `.jpg`, `.jpeg`, `.png`.
3. Recommended naming format:
   ```
   reference_images/
   ├── xray001.jpg
   ├── xray002.jpg
   ├── xray003.jpg
   └── ...
   ```

## IMPORTANT PRIVACY & ETHICAL GUIDELINES
- **DO NOT** store identifiable patient data, patient health information (PHI), or protected personal medical records in this folder.
- Only use permitted, de-identified academic benchmark images or synthetic test radiographs.
- The sample files provided here (`xray001.jpg` through `xray005.jpg` and `sample_query.jpg`) are purely synthetic demonstration radiographs created specifically for testing and college project demonstration.

## Updating Embeddings
Whenever you add, modify, or remove images from this folder, re-generate the precomputed feature embeddings by executing:
```bash
python scripts/generate_embeddings.py
```
This updates `embeddings/reference_embeddings.npz` so that the backend can instantaneously query the latest dataset without re-extracting DenseNet-121 features on every request.
