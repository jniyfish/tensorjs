let mode;
(async function () {
    console.log("loading");
    model = await tf.loadModel('http://localhost:81/tfjs-models/model/model.json');
    $('.progress-bar').hide();
    $('#loading').hide();
})();

$("#histogram-button").click(async function () {

    let img = await cv.imread('addnoise');

    let img_gray = new cv.Mat();
    cv.cvtColor(img, img_gray, cv.COLOR_BGR2GRAY, dstCn = 0)
    let img_blur = new cv.Mat();
    cv.medianBlur(img,img_blur,5);
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
    var dataset = Array.from(values);

 
    var width = 480;
    var height = 480;
    var padding = {
        top: 0,
        right: 8.2,    
        bottom: 50.2,   //50.2
        left: 18.4   
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
    var yAxisWidth = 480;
    var xTicks = hisData.map(function (d) {
        return d.x
    }) 

    var xScale = d3.scale.ordinal() 
        .domain(xTicks) 
        .rangeBands([0,xAxisWidth]) 

    var high=d3.max(hisData, function(d){ return d.y; })/dataset.length;
    var yScale = d3.scale.linear() 
        .domain([ 0, high])
        .range([0, 480*high*10.1]) 
        
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom")

    svg.append("g")
        .attr("class","axis")
        .attr("transform", "translate(" + padding.left + "," + (height - padding.bottom) + ")")
        
    console.log(xScale.range());
	console.log(xScale.rangeBand());
    
    var gRect = svg.append("g")
        .attr("transform", "translate(" + padding.left + "," + (-padding.bottom) + ")")
        .style("opacity",1.0);
        
    gRect.selectAll("rect")
        .data(hisData)
        .enter()
        .append("rect")
        .attr("class", "rect")
        .style("fill", "gray")
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
    gRect.style("opacity",1.0);

    var lineGenerator = d3.svg.line()
							.x(function(d){ return xScale(d.x); })
							.y(function(d){ return height - yScale(d.y); })
							.interpolate("basis");
	
	var gLine = svg.append("g")
                    .attr("transform","translate(" + padding.left + "," + ( -padding.bottom ) +  ")")
                    .style("color","gray")
                    .style("fill", "gray")
					.style("opacity",1);
	
	gLine.append("path")
		.attr("class","linePath")
		.attr("d",lineGenerator(hisData));
    //add white 
    
    var svg2 = document.querySelector("svg");
    var rect2 = document.querySelector("rect");
    var svgData = new XMLSerializer().serializeToString(svg2);
    var canvas2 = document.getElementById("histo");
    var svgSize = svg2.getBoundingClientRect();
    canvas2.width = svgSize.width * 1;
    canvas2.height = svgSize.height * 1;
    canvas2.style.width = svgSize.width;
    canvas2.style.height = svgSize.height;
    var ctx2 = canvas2.getContext("2d");
    ctx2.fillStyle = "white";
    ctx2.fillRect(0, 0, canvas2.width, canvas2.height);
    ctx2.scale(1, 1);
    var img2 = document.createElement("img");
    img2.setAttribute("src", "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgData))));
    img2.onload = async function() {
    await ctx2.drawImage(img2, 0, 0,480,480);
    }
    await $("svg").empty(); 
    $('#loading').hide();

    $('#inputDiv').fadeIn(3000);
	$('#hisDiv').fadeIn(3000);
	$('#noiseDiv').fadeIn(4000);
});
var pre;
$("#predict-button").click(async function () {
    
    removeSvg();
    
    await $('#loading').show();
    $('#hisDiv').fadeOut(2000);
    $('#preDiv').fadeIn(3000);

    let image = $('#histo').get(0);
    
    let tensor = await tf.fromPixels(image, 1)
        .toFloat()
        .expandDims(0);
    
    const b = tf.scalar(255);

    const values = tensor.dataSync();
    const dataset =Array.from(values);

    tensor = tensor.div(b);

    let predictions = await model.predict(tensor).data();

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
    classname=top[0].className;
    top.forEach(function (p) {
        $('#prediction-list').append(`<li>${p.className}: ${p.probability.toFixed(7)}</li>`);
    });
   
    await $("#denoise").click();

});


$("#denoise").click(async function () {



    if(classname=="snp"){
        let img = await cv.imread('snpImg');
        let deno = new cv.Mat();
        cv.medianBlur(img,deno,3);
        cv.imshow('denoisepic',deno);
    }
    else if (classname=="gaussian")
    {
    }
    else if(classname=="speckle")
    {
        let img = cv.imread('snpImg');
        let deno = new cv.Mat();
       // cv.cvtColor(img, img, cv.COLOR_RGBA2RGB, 0);
       // cv.bilateralFilter(img, deno, 3, 78, 78, cv.BORDER_DEFAULT);
       cv.medianBlur(img,deno,3);
        cv.imshow('denoisepic', deno);
    }

    $('#loading').hide();

    $('#deDiv').fadeIn(3000);

    
})
function calcpsnr(dst)
{
    let mse;
    let src = $('#picture').get(0);
    let dest = $('#denoisepic').get(0);
    let tensor_src = tf.fromPixels(src, 3) //img to matrix
        .toFloat();
    let tensor_dst = tf.fromPixels(dest,3)
        .toFloat();
    tensor_dst = tensor_dst.sub(tensor_src);
    tensor_dst = tensor_dst.square();
    let value = tensor_dst.sum().dataSync();
    mse = value[0] / (src.width*src.height*3);
    psnr = 10.0 * Math.log10((255 * 255)/ mse);   
    console.log(psnr);
    
}

function removeSvg() {
    var elem = document.getElementById('mysvg');
    elem.parentNode.removeChild(elem);
}