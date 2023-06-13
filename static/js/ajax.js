function init(){
        $.ajax({
            type: "GET",
            url: "/init-model"
        }).done(function (){
            changeButton();
        });
}

function changeButton(){
    $("#loadingModelButton").hide();
    $("#executeModelButton").show();
}

function showResponse(answer){
    $("#response").empty();
    let answerArray = answer.split("\n");
    for (let parag of answerArray){
        $("#response").append("<p>" + parag + "</p>");
    }
}

function executeModel(){
    $("#spinner").show();
    $.ajax({
        type: "POST",
        url: "/execute-model",
        data: $('form').serialize()
    }).done(function (response){
        $("#spinner").hide();
        showResponse(JSON.parse(response).generated_text);
        $("#responseDiv").show();
    });
}

$(document).ready(function(){
    $("#executeModelButton").hide();
    $("#spinner").hide();
    $("#responseDiv").hide();

    init();

    $("#executeModelButton").click(function (e){
        e.preventDefault();
        executeModel();
    });

});

