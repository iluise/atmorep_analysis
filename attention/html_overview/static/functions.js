var selectedBatch = 0;
var selectedHead = 0;

function setSelectedBatch(i) {
    selectedBatch = i;
}

function setSelectedHead(i) {
    selectedHead = i;
}

function showSubButtons(i) {
    document.getElementById('image-display').style.display = 'none';
    document.getElementById('sub-buttons-heads').style.display = 'none';
    document.getElementById('sub-buttons-batch').style.display = 'none';
    selectedBatch = i;    
    if (visualType === 'BATCH') {
    document.getElementById('sub-buttons-heads').style.display = 'flex';
    } else{
        document.getElementById('sub-buttons-batch').style.display = 'flex';
    }
}

function showImage(i) {
    setSelectedHead(i)
    refreshImage();
    document.getElementById('image-display').style.display = 'flex';
}

function refreshImage() {
    
    imageSrc = '/static/images/'
    if (concatType === 'Mean') {
        imageSrc += 'Combined_plot_'+visualType +'_' + model_id + '_' + field;
    } else {
        imageSrc += 'Combined_plot_'+ visualType+ '_' + model_id + '_ST_' + field;
    }   
    if (visualType === 'BATCH') {
        imageSrc += '_batch1_'+ (selectedBatch-1) + '_head_' + (selectedHead-1) + '.png';
    } else {
        imageSrc += '_batch1_'+ (selectedBatch-1) + '_batch2_' + (selectedHead-1) + '.png';
    }

    // imageSrc += '?' + new Date().getTime();
    document.getElementById('image').src = imageSrc;
    console.log('Concat type in refreshImage:', concatType);
    console.log('Image source:', imageSrc);
    document.getElementById('image-display').style.display = 'flex';
}

function updateModelID() {
    model_id = document.getElementById('model-type').value;
    // refreshImage();
    document.getElementById('image-display').style.display = 'none';
    document.getElementById('sub-buttons').style.display = 'none';
    console.log('model ID:', model_id);
}

function updateConcatType() {
    concatType = document.getElementById('concat-type').value;
    updatePage();
    console.log('Concat type in updateConcatType:', concatType);
}

function updateField() {
    field = document.getElementById('field-type').value;
    updatePage();
    console.log('field:', field);
}

function updateAttnType() {
    concatType = document.getElementById('attn-type').value;
    updatePage();
    console.log('Concat type in updateAttnType:', attnType);
}

function updateVisualType() {
    visualType = document.getElementById('visual-type').value;
    updatePage();
    console.log('Concat type in visual:', visualType);
}

function updatePage() {
    document.getElementById('image-display').style.display = 'none';
    document.getElementById('sub-buttons-heads').style.display = 'none';
    document.getElementById('sub-buttons-batch').style.display = 'none';
}

