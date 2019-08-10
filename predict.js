$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $('#selected-image').attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});

$("#histo-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $('#selected-histo').attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#histo-selector").prop('files')[0];
    reader.readAsDataURL(file);
});


var pre;
var imgg;

let mode;
(async function () {
    console.log("loading");
    model = await tf.loadModel('http://localhost:81/tfjs-models/model/model.json');
    $('.progress-bar').hide();
})();

$("#histogram-button").click(async function () {

    let ksize = new cv.Size(5, 5);

    let img = await cv.imread('selected-image');
    let src = await cv.imread('selected-image');


    let img_gray = new cv.Mat();
    cv.cvtColor(img, img_gray, cv.COLOR_BGR2GRAY, dstCn = 0)

    let img_blur = new cv.Mat();
    cv.blur(img, img_blur, ksize, anchor = new cv.Point(-1, -1), borderType = cv.BORDER_DEFAULT)

    let img_blur_gray = new cv.Mat();
    cv.cvtColor(img_blur, img_blur_gray, cv.COLOR_BGR2GRAY, dstCn = 0)

    cv.imshow('canvasOutput', img_gray);
    const b = tf.scalar(512);

    let image = $('#canvasOutput').get(0);
    let tensor_img_gray = tf.fromPixels(image, 1)
        .flatten()
        .toFloat()
        .expandDims(0);

    console.log(tensor_img_gray);

    cv.imshow('canvasOutput', img_blur_gray);

    image = $('#canvasOutput').get(0);
    let tensor_img_blur_gray = tf.fromPixels(image, 1)
        .flatten()
        .toFloat()
        .expandDims(0);

    tensor_img_blur_gray = tensor_img_blur_gray.div(b);
    tensor_img_gray = tensor_img_gray.div(b);


    tensor_img_gray = tensor_img_gray.sub(tensor_img_blur_gray);

    const values = tensor_img_gray.dataSync();
    const dataset = Array.from(values);
    console.log(dataset);
    ////////
    var width = 480;
    var height = 480;
    var padding = {
        top: 0,
        right: 25,
        bottom: 51,
        left: 25
    };

    var svg = d3.select("body").append("svg")
        .attr("id","mysvg")
        .attr("width", width)
        .attr("height", height);

    var binNum = 512,
        rangeMin = -0.5,
        rangeMax = 0.5;

    var histogram = d3.layout.histogram()
        .range([rangeMin, rangeMax])
        .bins(binNum)
        .frequency(false);

    var hisData = histogram(dataset);

    console.log(hisData);

    var xAxisWidth = width - padding.left - padding.right
    var yAxisWidth = 480
    var xTicks = hisData.map(function (d) {
        return d.x
    }) // ???layout?????????x

    var xScale = d3.scale.ordinal() // ?????
        .domain(xTicks) // ???
        .rangeBands([0, xAxisWidth]) // ??

    var yScale = d3.scale.linear() // ?????
        .domain([0,0.07])
        .range([0.1, yAxisWidth]) // ??

    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom")

    svg.append("g")
        .attr("transform", "translate(" + padding.left + "," + (height - padding.bottom) + ")")

    var gRect = svg.append("g")
        .attr("transform", "translate(" + padding.left + "," + (-padding.bottom) + ")")

    gRect.selectAll("rect")
        .data(hisData)
        .enter()
        .append("rect")
        .attr("class", "rect")
        .style("fill", "#ff0000")
        .attr("x", function (d, i) {
            return xScale(d.x)
        })
        .attr("y", function (d, i) {
            return height - yScale(d.y)
        })
        .attr("width", function (d, i) {
            return xScale.rangeBand()
        })
        .attr("height", function (d) {
            return yScale(d.y)
        }
        )

    ///////////////////////////
    gRect.style("opacity",0.0);

    var lineGenerator = d3.svg.line()
							.x(function(d){ return xScale(d.x); })
							.y(function(d){ return height - yScale(d.y); })
							.interpolate("basis");
	
	var gLine = svg.append("g")
                    .attr("transform","translate(" + padding.left + "," + ( -padding.bottom ) +  ")")
                    .style("color","red")
                    .style("fill", "#ff0000")
					.style("opacity",1);
	
	gLine.append("path")
		.attr("class","linePath")
		.attr("d",lineGenerator(hisData));




});

$("#predict-button").click(async function () {
    let image = $('#selected-histo').get(0);
    let tensor = tf.fromPixels(image, 1)
        .toFloat()
        .expandDims(0);

    const b = tf.scalar(255);

    const values = tensor.dataSync();
    const dataset = Array.from(values);
    console.log(dataset);
    tensor = tensor.div(b);

    let predictions = await model.predict(tensor).data();
    console.log(predictions); //tensor is predict target matrix
    pre=predictions;

    let top = Array.from(predictions)
        .map(function (p, i) {
            return {
                probability: p,
                className: IMAGENET_CLASSES[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0, 5);

    $("#prediction-list").empty();
    top.forEach(function (p) {
        $('#prediction-list').append(`<li>${p.className}: ${p.probability.toFixed(7)}</li>`);
    });



});

///////////
$("#save").click(async function () {
    $(".svg").empty();
    var svg = document.querySelector("svg");
    var rect = document.querySelector("rect")
    var svgData = new XMLSerializer().serializeToString(svg);
    var canvas = document.createElement("canvas");
    var svgSize = svg.getBoundingClientRect();
    canvas.width = svgSize.width * 1;
    canvas.height = svgSize.height * 1;
    canvas.style.width = svgSize.width;
    canvas.style.height = svgSize.height;
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.scale(1, 1);
    var img = document.createElement("img");
    img.setAttribute("src", "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData))));
    img.onload = function() {
      ctx.drawImage(img, 0, 0,480,480);
     
      var canvasdata = canvas.toDataURL("image/jpg", 1);
  
      var pngimg = '<img src="' + canvasdata + '">';
      d3.select("#pngdataurl").html(pngimg);
  
      var a = document.createElement("a");
      a.download = "histogram" + ".jpg";
      a.href = canvasdata;
      document.body.appendChild(a);
      a.click();
    };
  })

$("#denoise").click(async function () {

    var maxIndex=3;
    for(let i=0,max=0;i<3;i++)
    {
        if(pre[i]>max)
        {
            max=pre[i];
            maxIndex=i;
        }
    }
    console.log(maxIndex);
                            // 0:'snp',   medianblur
                            //1:'gaussian', //nonopolmean
                            //2:'speckle'  double side filter
  

    if(maxIndex==0)  //case snp
    {
        let img = await cv.imread('selected-image');
        let denoise = new cv.Mat();
        cv.medianBlur(img, denoise, 3);
        cv.imshow('Result', denoise);
    }
    else if(maxIndex==2)  //case spe
    {
        let img = await cv.imread('selected-image');
        let denoise = new cv.Mat();
        cv.cvtColor(img, img, cv.COLOR_RGBA2RGB, 0);
        cv.bilateralFilter(img, denoise, 3, 78, 78, cv.BORDER_DEFAULT);
        cv.imshow('Result', denoise);
    }
});