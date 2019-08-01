$("#image-selector").change(function(){
    let reader = new FileReader();
    reader.onload = function()
    {
        let dataURL = reader.result;
        $('#selected-image').attr("src",dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});




let mode;
(async function(){
    console.log("loading");
    model =await tf.loadModel('http://localhost:81/tfjs-models/model/model.json');
    $('.progress-bar').hide();
})();

$("#predict-button").click(async function(){
    let image = $('#selected-image').get(0);
    console.log(image);
    let tensor = tf.fromPixels(image,1)
        .resizeNearestNeighbor([480,480])   
        .toFloat()
        .expandDims(0);

    const b = tf.scalar(255);

    tensor = tensor.div(b);
        //preproces to be add

    let predictions = await model.predict(tensor).data(); 
    console.log(predictions);  //tensor is predict target matrix

    let top = Array.from(predictions)
        .map(function(p,i){
            return{
                probability: p,
                className:IMAGENET_CLASSES[i]
            };
        }).sort(function(a,b){
            return b.probability - a.probability;
        }).slice(0,5);
    
    $("#prediction-list").empty();
    top.forEach(function (p) {
        $('#prediction-list').append(`<li>${p.className}: ${p.probability.toFixed(7)}</li>`);
    });
    
});

