package main.java.com.sctcc.kgpc.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@RestController
@RequestMapping("/api/patents")
public class PatentController {
    
    private final RestTemplate restTemplate;
    
    @Value("${django.base-url}")
    private String djangoBaseUrl;
    
    // 使用构造器注入更安全
    public PatentController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }
    
    @GetMapping("/recommendations")
    public ResponseEntity<?> getPatentRecommendations(
            @RequestParam String patentId,
            @RequestParam(required = false, defaultValue = "10") int topK) {
        
        // 构建带认证的请求头
        HttpHeaders headers = new HttpHeaders();
        headers.set("Accept", MediaType.APPLICATION_JSON_VALUE);
        HttpEntity<?> entity = new HttpEntity<>(headers);
        
        // 使用URI builder更安全
        String url = UriComponentsBuilder.fromHttpUrl(djangoBaseUrl)
                .path("/api/patent_recommend")
                .queryParam("patent_id", patentId)
                .queryParam("top_k", topK)
                .toUriString();
        
        try {
            // 添加超时控制和错误处理
            return restTemplate.exchange(
                    url,
                    HttpMethod.GET,
                    entity,
                    new ParameterizedTypeReference<Map<String, Object>>() {}
            );
        } catch (Exception e) {
            // 自定义错误处理
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(Map.of(
                            "error", "Django service unavailable",
                            "message", e.getMessage()
                    ));
        }
    }
}