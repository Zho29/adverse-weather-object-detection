#!/bin/bash
# Create YOLO dataset structure for high fog (beta=0.02)
# Reuses labels from beta=0.01, points to beta=0.02 foggy images

set -e

echo "======================================================================"
echo "Creating YOLO Dataset for High Fog (Beta=0.02)"
echo "======================================================================"

SRC_LABELS="/mnt/Data/as19197/new_project/yolo_dataset_beta0.01_restored/labels"
DEST_ROOT="/mnt/Data/as19197/new_project/yolo_dataset_beta0.02"
FOGGY_ROOT="/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy"

echo "Source labels: $SRC_LABELS"
echo "Destination: $DEST_ROOT"
echo "Foggy images: $FOGGY_ROOT"

# Create directory structure
mkdir -p "$DEST_ROOT"/{images,labels}/{train,val,test}

echo ""
echo "[1/3] Copying labels from beta=0.01 dataset..."
cp -r "$SRC_LABELS"/* "$DEST_ROOT/labels/"
echo "✓ Labels copied"

echo ""
echo "[2/3] Linking foggy images (beta=0.02)..."
# Link train, val, test foggy images (beta=0.02)
for split in train val test; do
    echo "  Processing $split..."
    find "$FOGGY_ROOT/$split" -name "*_beta_0.02.png" | while read img; do
        ln -sf "$img" "$DEST_ROOT/images/$split/$(basename $img)"
    done
done
echo "✓ Images linked"

echo ""
echo "[3/3] Creating data.yaml..."
cat > "$DEST_ROOT/data.yaml" << EOF
# Foggy Cityscapes YOLO Dataset - High Fog (Beta=0.02)
path: $DEST_ROOT
train: images/train
val: images/val
test: images/test

# Cityscapes classes (8 classes)
names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle

nc: 8
EOF
echo "✓ data.yaml created"

echo ""
echo "======================================================================"
echo "Dataset Summary:"
echo "======================================================================"
echo "Train images: $(find $DEST_ROOT/images/train -name "*.png" 2>/dev/null | wc -l)"
echo "Val images:   $(find $DEST_ROOT/images/val -name "*.png" 2>/dev/null | wc -l)"
echo "Test images:  $(find $DEST_ROOT/images/test -name "*.png" 2>/dev/null | wc -l)"
echo ""
echo "Train labels: $(find $DEST_ROOT/labels/train -name "*.txt" 2>/dev/null | wc -l)"
echo "Val labels:   $(find $DEST_ROOT/labels/val -name "*.txt" 2>/dev/null | wc -l)"
echo "Test labels:  $(find $DEST_ROOT/labels/test -name "*.txt" 2>/dev/null | wc -l)"
echo "======================================================================"
echo "✓ YOLO dataset ready at: $DEST_ROOT"
echo "======================================================================"
