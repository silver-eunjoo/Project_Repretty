import kotlinx.coroutines.*
import model.NaverModel
import org.dhatim.fastexcel.Workbook
import org.jsoup.Connection
import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.SocketTimeoutException
import java.util.Collections

const val LIMIT_URL_FIND = 15 //15페이지당 제한
const val LIMIT_PARSE = 15 //15번째마다 제한
const val DELAY : Long = 5000 //5초정도 딜레이

fun main(args: Array<String>) {
    val targetUrl =
        "https://kin.naver.com/search/list.naver?query=%EA%B0%B1%EB%85%84%EA%B8%B0&section=qna&dirId=7011404"
    val startPage = 1
    val crawlAmount = 20
    val targetFile = File("${System.getProperty("user.dir")}\\KINCrawl\\result.xlsx")
    val urls = parseUrls(targetUrl, crawlAmount, startPage)
    runBlocking {
        val models = parseModels(urls)
        writeExcel(models, targetFile)
    }
}

fun writeExcel(models: List<NaverModel>, file: File) {
    val os = ByteArrayOutputStream()
    val workbook = Workbook(os, "Naver Data", "1.0")
    val sheet = workbook.newWorksheet("data")
    sheet.apply {
        value(0, 0, "번호")
        value(0, 1, "제목")
        value(0, 2, "질문")
        value(0, 3, "내용")
        value(0, 4, "url")

    }
    for ((index, model) in models.withIndex()) {
        //한 행당
        sheet.value(index + 1, 0, index + 1) //1줄 띄우고 (개체 타입)
        sheet.value(index + 1, 1, model.title)
        sheet.value(index + 1, 2, model.question)
        sheet.value(index + 1, 3, model.answer)
        sheet.value(index + 1, 4, model.url)
    }
    workbook.finish()
    os.writeTo(FileOutputStream(file))
    os.close()
}

fun parseUrls(targetUrl: String, crawlAmount: Int, startPage: Int): List<String> {
    var page = startPage//시작 페이지
    val urlList: MutableList<String> = MutableList(0) { _ -> "" } //url 저장용
    while (true) {
        val connection: Connection = Jsoup.connect("$targetUrl&page=$page").apply {
            ignoreContentType(false)
            userAgent("Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:25.0) Gecko/20100101 Firefox/25.0")
            referrer("https://www.naver.com")
            timeout(5000)
            followRedirects(true)
        }
        val doc: Document = connection.get() //get요청
        val liList = doc.getElementsByClass("basic1")[0].getElementsByTag("li") //Q&A li list
        liList.forEach { li ->
            //li list 반복
            if (crawlAmount > urlList.size) {
                val dt = li.getElementsByTag("dl")[0].getElementsByTag("dt")[0] //질문 들어있는 dt블록 가져옴
                if (dt.html().contains("ico_pro")) { //전문가 태그 달려있는 경우 가져옴
                    val img = dt.getElementsByTag("img")
                    //의사 답변인경우
                    if (img.attr("alt") == "의사답변") {
                        //해당 태그랑 같이 있는 a 태그에서 href 가져와서 리스트 추가
                        val link = dt.getElementsByTag("a")[0].attr("href")
                        urlList.add(link) //링크 추가
                    }
                }
            }
        }
        println("current page : $page")
        println("crawled count : ${urlList.size} / $crawlAmount")
        if (page % LIMIT_URL_FIND == 0 && page != 0) {
            //딜레이 넣어 네이버 제한 우회
            println("delay in ${DELAY / 1000}s")
            Thread.sleep(DELAY)
        }
        //다음페이지가 없을경우 중단 or 충분한 데이터 모아졌으면
        if (!doc.html().contains("다음페이지") || crawlAmount <= urlList.size)
            break
        else
            page++
    }
    return urlList
}

suspend fun parseModels(urls: List<String>): List<NaverModel> {
    val models: MutableList<NaverModel> =
        Collections.synchronizedList(mutableListOf()) //코루틴도 동시성 문제 발생위험 -> 추가시 sync된 리스트 사용
    withContext(Dispatchers.IO) {
        //IO 콘텍스트 변경
        for ((index, url) in urls.withIndex()) {
            if (index % LIMIT_PARSE == 0) {
                //딜레이 넣어 네이버 제한 우회
                println("delay in ${DELAY / 1000}s")
                delay(DELAY)
            }
            launch {
                //파싱중 오류가 없다면 add
                parseModel(url)?.let { models.add(it) }
            }
        }
    }
    return models
}

fun parseModel(url: String): NaverModel? {
    println("parsing url - $url")
    try {
        val doc: Document = Jsoup.connect(url).apply {
            ignoreContentType(false)
            userAgent("Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:25.0) Gecko/20100101 Firefox/25.0")
            referrer("https://www.naver.com")
            timeout(3000)
            followRedirects(true)
        }.get() //get요청
        val title: String = doc.getElementsByClass("title").text()
        val question: String = doc.getElementsByClass("c-heading__content").text()
        val answer: String = doc.getElementsByClass("_endContentsText c-heading-answer__content-user").text() //채택 여부와 상관없이 첫번째 질문 흭득
        return NaverModel(title, question, answer, url)
    } catch (e: SocketTimeoutException) {
        println("Timed out - $url")
        return null
    }
}