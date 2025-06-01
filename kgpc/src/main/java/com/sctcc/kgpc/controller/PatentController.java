package com.sctcc.kgpc.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@RestController
public class PatentController {
    @Value("${django.ip}")
    private String djangoIp; // 从配置文件中读取 Django 服务的 IP 地址
    private final RestTemplate restTemplate = new RestTemplate();

    @GetMapping("/patent_recommend")
    public ResponseEntity<?> patentRecommend(@RequestParam(name = "patent_id") String patentId) {
        String url = "http://" + djangoIp + ":8000/api/patent_recommend"; // Django 服务的完整 URL
        // 使用 UriComponentsBuilder 构建带参数的 URL
        UriComponentsBuilder builder = UriComponentsBuilder.fromHttpUrl(url)
                .queryParam("patent_id", patentId);
        // 执行 GET 请求
        ResponseEntity<String> responseEntity = restTemplate.exchange(
                builder.toUriString(), // 使用构建好的 URL
                HttpMethod.GET,
                null, // 使用 null 代表不发送任何请求
                new ParameterizedTypeReference<String>() {} // 指定响应体类型为 String
        );
        return responseEntity;
    }
}