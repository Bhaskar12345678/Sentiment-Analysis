< !DOCTYPE html >
    <
    html >

    <
    head >

    <
    meta charset = "utf-8" >
    <
    meta name = "viewport"
content = "width=device-width, initial-scale=1, shrink-to-fit=no" >
    <
    title > Sentiment Analysis < /title>

<!-- Bootstrap CSS -->
<
link rel = "stylesheet"
href = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
integrity = "sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm"
crossorigin = "anonymous" >



    <
    /head>

<
body >

    <
    link rel = "stylesheet"
type = "text/css"
href = "main.css" >

    <
    div style = "text-align:center;border:1px solid red;background-color:lightblue" >
    <
    table class = "blueTable" >
    <
    thead >
    <
    tr >
    <
    th >
    <
    h3 > Unsupervised Sentiment Analysis < /h3> < /
th >

    <
    /tr> <
tr >
    <
    th >
    <
    Form >
    <
    button class = "btn-primary"
onclick = "LoadReviews()" > Load Reviews... < /button>

<
/Form> < /
th >

    <
    /tr> < /
thead > <
    /table> < /
div >


    res.render('1.ejs', {
        reviews: nextPageReviews,
        currentPage: page,
        pages: Math.ceil(totalnumOfReviews / resPerPage),
        //  searchVal: searchQuery, 
        numOfReviews: numOfReviews
    });


<% if (nextPageReviews.length) { %>
<% nextPageReviews.forEach(function(review) { %> <
tr >
    <
    td >
    <%= review.review %> <
    /td>  <
    td >
    <%= review.sentiment %> <
    /td>  <
    /tr>
<% }) %>
<% } %>



<
div id = "accordion" >
    <
    div class = "card" >
    <
    div class = "card-header"
id = "headingOne" >
    <
    h5 class = "mb-0" >
    <
    button class = "btn btn-link"
data - toggle = "collapse"
data - target = "#collapseOne"
aria - expanded = "true"
aria - controls = "collapseOne" >
    Collapsible Group Item #1















              </button>















                </h5>















            </div>































            <div id= "collapseOne"
class = "collapse show"
aria - labelledby = "headingOne"
data - parent = "#accordion" >
    <
    div class = "card-body" >
    Anim pariatur cliche reprehenderit, enim eiusmod high life accusamus terry richardson ad squid .3 wolf moon officia aute, non cupidatat skateboard dolor brunch.Food truck quinoa nesciunt laborum eiusmod.Brunch 3 wolf moon tempor, sunt aliqua put a bird
on it squid single - origin coffee nulla assumenda shoreditch et.Nihil anim keffiyeh helvetica, craft beer labore wes anderson cred nesciunt sapiente ea proident.Ad vegan excepteur butcher vice lomo.Leggings occaecat craft beer farm - to - table,
    raw denim aesthetic synth nesciunt you probably haven 't heard of them accusamus labore sustainable VHS. < /
div > <
    /div> < /
div > <
    div class = "card" >
    <
    div class = "card-header"
id = "headingTwo" >
    <
    h5 class = "mb-0" >
    <
    button class = "btn btn-link collapsed"
data - toggle = "collapse"
data - target = "#collapseTwo"
aria - expanded = "false"
aria - controls = "collapseTwo" >
    Collapsible Group Item #2















              </button>















                </h5>















            </div>















            <div id= "collapseTwo"
class = "collapse"
aria - labelledby = "headingTwo"
data - parent = "#accordion" >
    <
    div class = "card-body" >
    Anim pariatur cliche reprehenderit, enim eiusmod high life accusamus terry richardson ad squid .3 wolf moon officia aute, non cupidatat skateboard dolor brunch.Food truck quinoa nesciunt laborum eiusmod.Brunch 3 wolf moon tempor, sunt aliqua put a bird
on it squid single - origin coffee nulla assumenda shoreditch et.Nihil anim keffiyeh helvetica, craft beer labore wes anderson cred nesciunt sapiente ea proident.Ad vegan excepteur butcher vice lomo.Leggings occaecat craft beer farm - to - table,
    raw denim aesthetic synth nesciunt you probably haven 't heard of them accusamus labore sustainable VHS. < /
div > <
    /div> < /
div > <
    div class = "card" >
    <
    div class = "card-header"
id = "headingThree" >
    <
    h5 class = "mb-0" >
    <
    button class = "btn btn-link collapsed"
data - toggle = "collapse"
data - target = "#collapseThree"
aria - expanded = "false"
aria - controls = "collapseThree" >
    Collapsible Group Item #3















              </button>















                </h5>















            </div>















            <div id= "collapseThree"
class = "collapse"
aria - labelledby = "headingThree"
data - parent = "#accordion" >
    <
    div class = "card-body" >
    Anim pariatur cliche reprehenderit, enim eiusmod high life accusamus terry richardson ad squid .3 wolf moon officia aute, non cupidatat skateboard dolor brunch.Food truck quinoa nesciunt laborum eiusmod.Brunch 3 wolf moon tempor, sunt aliqua put a bird
on it squid single - origin coffee nulla assumenda shoreditch et.Nihil anim keffiyeh helvetica, craft beer labore wes anderson cred nesciunt sapiente ea proident.Ad vegan excepteur butcher vice lomo.Leggings occaecat craft beer farm - to - table,
    raw denim aesthetic synth nesciunt you probably haven 't heard of them accusamus labore sustainable VHS. < /
div > <
    /div> < /
div > <
    /div>



<!-- Optional JavaScript -->
<!-- jQuery first, then Popper.js, then Bootstrap JS -->
<
script src = "https://code.jquery.com/jquery-3.2.1.slim.min.js"
integrity = "sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN"
crossorigin = "anonymous" > < /script> <
script src = "https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"
integrity = "sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q"
crossorigin = "anonymous" > < /script> <
script src = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"
integrity = "sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl"
crossorigin = "anonymous" > < /script>






<
/body>

<
/html>