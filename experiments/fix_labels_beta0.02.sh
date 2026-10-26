#!/bin/bash
# Fix label filenames to match restored image filenames

set -e

RESTORED_DIR="/mnt/Data/as19197/new_project/yolo_dataset_beta0.02_restored"

echo "======================================================================"
echo "Fixing Label Filenames for Beta=0.02 Restored Dataset"
echo "======================================================================"

for split in train val test; do
    echo ""
    echo "Processing $split..."
    
    img_dir="$RESTORED_DIR/images/$split"
    label_dir="$RESTORED_DIR/labels/$split"
    
    # Create temp directory for new labels
    temp_dir="$RESTORED_DIR/labels/${split}_new"
    mkdir -p "$temp_dir"
    
    # For each image, create corresponding label
    count=0
    for img in "$img_dir"/*.png; do
        if [ -f "$img" ]; then
            # Get image basename without extension
            img_base=$(basename "$img" .png)
            
            # Extract the city and frame info (e.g., frankfurt_000000_000294)
            city_frame=$(echo "$img_base" | sed 's/_leftImg8bit_foggy_beta_0.02//')
            
            # Find matching label (old format)
            old_label=$(find "$label_dir" -name "${city_frame}_*_beta0.01.txt" -o -name "${city_frame}_beta0.01.txt" | head -1)
            
            if [ -f "$old_label" ]; then
                # Copy with new name matching image
                cp "$old_label" "$temp_dir/${img_base}.txt"
                ((count++))
            fi
        fi
    done
    
    # Replace old labels with new ones
    rm -rf "$label_dir"
    mv "$temp_dir" "$label_dir"
    
    echo "  ✓ Fixed $count labels in $split"
done

echo ""
echo "======================================================================"
echo "Verification:"
echo "======================================================================"
for split in train val test; do
    img_count=$(find "$RESTORED_DIR/images/$split" -name "*.png" | wc -l)
    label_count=$(find "$RESTORED_DIR/labels/$split" -name "*.txt" | wc -l)
    echo "$split: $img_count images, $label_count labels"
done
echo "======================================================================"
echo "✓ Label filenames fixed!"
echo "======================================================================"
