$(document).ready(function () {
  // Remove gridlines 
  Chart.defaults.scale.gridLines.display = false;

  //Load CSV
  d3.csv("../data/HAM10000_metadata_cleaned.csv", function (d) {
    return {
      age: +d.age,
      sex: d.sex,
      localization: d.localization,
      dxType: d.dx_type,
      cellType: d.cell_type

    }
  }).then(function (data) {
    let ctx = $('#cellTypeChart')[0].getContext('2d')

    let groupedData = sort(_.chain(data).countBy('cellType').value())

    let chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: _.pluck(groupedData, 'label'),
        datasets: [{
          data: _.pluck(groupedData, 'value'),
          backgroundColor: '#9e768f',
          borderColor: 'black'
        }]

      },
      options: {
        sort: true,
        legend: {
          display: false
        },
        title: {
          display: false
        },
        scales: {xAxes: [{
          ticks: {
            callback: function (value) {
              return decamelize(value)
            }
          }
        }],
    
        yAxes: [{
          ticks: {
            beginAtZero: true
          }
        }]}
      }
    });
  });

});

function decamelize(str) {
  var words = str.match(/[A-Za-z][a-z]*/g);
  return words.map(capitalize).join(' ');
}

function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.substring(1)
}

function sort(obj) {
  let arr = []

  _.mapObject(obj, function(val, key) {
    arr.push({
      label: key,
      value: val
    })
  });

  return _.sortBy(arr, 'value')
}