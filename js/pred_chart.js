var color_red = 'rgb(255, 99, 132)';
var color_blue = 'rgb(54, 162, 235)';

/******************************** 
*
*********************************/
function get_chart_config(items, pred_itrems){
    var config = {
        type: 'line',
        data: {
            labels: items.lbl ,
            datasets: [
                {
                    label: 'H-num',
                    fill: false,
                    backgroundColor: color_red,
                    borderColor: color_red,
                    data: items.hnum ,
                },
                {
                    label: 'predict',
                    fill: false,
                    backgroundColor: color_blue,
                    borderColor: color_blue,
                    data: pred_itrems ,
                }
            ]
        },
        options: {
            responsive: true,
            title: {
                display: true,
                text: ' '
            },
            tooltips: {
                mode: 'index',
                intersect: false,
            },
            hover: {
                mode: 'nearest',
                intersect: true
            },
            scales: {
                xAxes: [{
                display: true,  
                    ticks: {
                        autoSkip: false,
                    },
                }],
                yAxes: [{
                    display: true,
                    scaleLabel: {
                        display: true,
                        labelString: 'Value'
                    }
                }]
            }
        }
    };
    return config;
}
/******************************** 
*
*********************************/
function convert_chart_arr( items ){
	var hnum = []
	var lbl = []
	items.forEach( function (item) {
//console.log( item );    
		lbl.push( item.no )                
		hnum.push( item.hnum )                
	});
	var ret= {
		"lbl" : lbl,
		"hnum" : hnum,
	}
	return ret;
}