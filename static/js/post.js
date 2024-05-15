$(function () {
    var file = null;
    var blob = null;

    $("input[type=file]").change(function () {
        file = $(this).prop("files")[0];

        // ファイルチェック
        if (!file.name.endsWith(".nii")) {
            file = null;
            blob = null;
            alert("niiファイルを選択してください");
            return; // ファイルが適切でない場合、ここで処理を終了
        }

        // ファイルが適切な場合、Blobを作成
        blob = new Blob([file], { type: "application/octet-stream" });
    });
  
    // アップロード開始ボタンがクリックされたら
    $("#post").click(function () {
        if (!file || !blob) {
            return; // ファイルまたはBlobがなければ何もしない
        }
        var fd = new FormData();
        fd.append("files", blob, file.name);

        // API宛にPOSTする
        $.ajax({
            url: "/api/image_recognition",
            type: "POST",
            dataType: "json",
            data: fd,
            processData: false,
            contentType: false,
        })
        .done(function (data, textStatus, jqXHR) {
            // 通信が成功した場合の処理
            console.log(data);
            // var result = document.getElementById("result");
            const urls = data.urls.join(',');
            window.location.href = `/clinical-info?urls=${encodeURIComponent(urls)}`;
            // window.location.href = '/clinical-info';  // clinical-infoに遷移
        })
        .fail(function (jqXHR, textStatus, errorThrown) {
            // 通信が失敗した場合のエラー出力
            var result = document.getElementById("result");
            result.innerHTML = "サーバーとの通信が失敗した...";
        });
    });
});
  
  