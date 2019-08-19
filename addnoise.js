$("#addAll").click(async function () {

    await gauFunction();
   
    await sleep(2000);
    await speFunction();

    await sleep(2000);
    await snpFunction();
})

$("#showAll").click(async function()
{
    $('#deDiv').hide();
    $('#noiseDiv').hide();
    $('#preDiv').hide();
    $('.show').show();
})