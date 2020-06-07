const fs = require('fs');
const express = require('express');
const router = express.Router();
var cors = require('cors')
const bodyParser = require('body-parser');
const request = require('request');
const app = express();

const csv = require('csv-parser');

app.use(cors())

app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true }));
app.set('view engine', 'ejs')


// Read all the reviews from movie_review.csv file into a global array to traverse 
var reviews = [];
var sentiments = [];
var visitOnce = 0;
var currentPage = 0;

// To Do - Need to remove special HTML charcaters from the text before processing
fs.createReadStream('movie_reviews.csv')
    .pipe(csv())
    .on('data', (data) => reviews.push(data))
    .on('end', () => {
        console.log(reviews)
        console.log('CSV file successfully processed');
    });


// On Site load call the following route
app.get('/', async(req, res, next) => {

    console.log('second time call ..............................................................');
    if (visitOnce == 0) {
        visitOnce = 1;
        currentPage = 1;
        res.render('new1', { pageNo: 1 });
    } else {
        console.log('second time call .............................................................. Current Page', currentPage);
        //res.render('new1', { pageNo: currentPage });
    }
})

// Get the first/next/last/previous page functinality router
app.get('/reviews/:page', async(req, res, next) => {

    console.log('inside reviews route-------------------------------------------------------------- :       ', req.params.page);

    // Declaring variables
    const resPerPage = 10; // results per page
    const page = req.params.page || 1; // Page 
    currentPage = page;
    try {
        let nextPageReviews = [];

        let icount = 0;
        for (let index = ((resPerPage * page) - resPerPage); index < (resPerPage * page); index++) {
            console.log('index ----', index);
            nextPageReviews.push(reviews[index]);
            console.log(nextPageReviews[icount])
            icount++;
        }

        const numOfReviews = nextPageReviews.length;
        const totalnumOfReviews = reviews.length;

        console.log('total reviews count: ', totalnumOfReviews);
        console.log('review count for the current page: ', numOfReviews);

        res.send(nextPageReviews);

    } catch (err) {
        throw new Error(err);
    }
});

/*
var uint8arrayToString = function(data) {
    return String.fromCharCode.apply(null, data);
};
*/

// Get sentiment route - called when getSentiment button will be clicked for a given review with selected algo on a selected page
app.post('/11', async(req, res, next) => {
    console.log('Inside get Sentiment route -----------------', req.body.reviewRecord);
    console.log(req.body.pageNo);
    console.log(req.body.algo);
    const resPerPage = 10; // results per page

    var reviewid = parseInt(req.body.reviewRecord);
    var pageNo = parseInt(req.body.pageNo) - 1;
    var algo = req.body.algo;

    // Fetch the correct review from the array list based on passed parameter PageNo and Reviewid
    reviewText = reviews[reviewid + (pageNo * resPerPage) - 1];

    console.log('------------------------------------      Getting the right record from global array              ------------------------------------------------------------');
    console.log('Record id is :', (pageNo * resPerPage) + reviewid - 1);
    console.log(reviewText.review);
    console.log('-------------------------------------  sending data to Python program   -------------------------------------------------------')




    // call Python program for all Unsupervised, Supervised and LSTM models
    const spawn = require("child_process").spawn;
    let pythonProcess = null;

    if (algo == 'Afinn') {
        pythonProcess = spawn(`c:/Anacondo3/envs/chatbot/python`, ['C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/Unsupervised_Predict_sentiment_Afinn_model.py', reviewText.review]);
    }
    if (algo == 'Sentiwordnet') {
        console.log('Inside Sentiwordnet lexicon');
        pythonProcess = spawn(`c:/Anacondo3/envs/chatbot/python`, ['C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/Unsupervised_Predict_sentiment_Sentiwordnet_lexicon.py', reviewText.review]);
    }
    if (algo == 'Vader') {
        console.log('Inside Vader lexicon');
        pythonProcess = spawn(`c:/Anacondo3/envs/chatbot/python`, ['C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/Unsupervised_Predict_sentiment_Vader_lexicon.py', reviewText.review]);
    }
    if (algo == 'SGD') {
        console.log('Inside SGD lexicon');
        pythonProcess = spawn(`c:/Anacondo3/envs/chatbot/python`, ['C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/Supervised_Predict_sentiment_SGDClassifier_model.py', reviewText.review]);
    }
    if (algo == 'LogisticReg') {
        pythonProcess = spawn(`c:/Anacondo3/envs/chatbot/python`, ['C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/Supervised_Predict_sentiment_LogisticRegressionClassifier_model.py', reviewText.review]);
    }
    if (algo == 'NaiveBias') {
        pythonProcess = spawn(`c:/Anacondo3/envs/chatbot/python`, ['C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/Supervised_Predict_sentiment_MultinomialNB_Classifier_model.py', reviewText.review]);
    }
    if (algo == 'LSTM') {
        pythonProcess = spawn(`c:/Anacondo3/envs/chatbot/python`, ['C:/Bhaskar Sem 8/Capstone project/Capstome Sem 8 Documents submitted to College/Sentiment_Analysis/Predict_Sentiment_LSTM_model.py', reviewText.review]);
    }

    console.log('-------------------------------------   After data review was sent       -------------------------------------------------------')
        /*Here we are saying that every time our node application receives data from the python process output stream(on 'data'), we want to convert that received data into a string and append it to the overall dataString.*/


    pythonProcess.stdout.on('data', function(data) {

        console.log('Output from Python : ');
        console.log(data.toString());

        res.send(data.toString());
        res.end();

    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`child stderr:\n${data}`);
        //let resbody = String({ "sentiment": +'Neutral' });
        //res.send(resbody);
    });


});


app.listen(3000, function() {
    console.log('App listening on port 3000!')

});