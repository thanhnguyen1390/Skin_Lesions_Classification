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
        scales: {
          xAxes: [{
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
          }]
        }
      }
    });
  });

  // //Load confusion matrix
  // d3.csv('../models/alexnet/test_result.csv').then(function (data) {
  //   // let confusion_matrix = calculate_confusion_matrix(data)
  //   let confusion_matrix = [[10, 189],[98,129]]
  //   labels = ['Actinic keratoses', 'Basal call carcinoma', 'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']
  //   Matrix({
  //     container: '#alexNetCM',
  //     data: confusion_matrix,
  //     labels: labels,
  //     start_color : '#ffffff',
  //     end_color : '#e67e22'
  //   })
  // })

});

function calculate_confusion_matrix(data) {
  let confusion_matrix = zeros(7, 7, 0)
  _.each(data, (row) => {
    confusion_matrix[row.Actual][row.Predicted] += 1
  })

  return confusion_matrix
}

function zeros(w, h) {
  return Array.from(new Array(h), _ => Array(w).fill(0))
}

function decamelize(str) {
  var words = str.match(/[A-Za-z][a-z]*/g);
  return words.map(capitalize).join(' ');
}

function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.substring(1)
}

function sort(obj) {
  let arr = []

  _.mapObject(obj, function (val, key) {
    arr.push({
      label: key,
      value: val
    })
  });

  return _.sortBy(arr, 'value')
}